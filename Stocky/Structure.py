import pathlib
import pandas as pd
import numpy as np
import math
import datetime
import itertools
import pickle
from collections import OrderedDict
import ipywidgets as widgets
from IPython.display import clear_output

from .Dataset import General

pd.options.display.max_columns = 30
transactions_tax = 0.3e-2
processing_fee = 0.1425e-2

class namedtuple:
    def __init__(self,**kwdarg):
        object.__setattr__(self,'dict',kwdarg)
    def __getattr__(self,key):
        return self.dict[key]
    def __setattr__(self,key,value):
        collect = self.dict
        assert collect.get(key,'not defined') is not 'not defined'
        collect[key] = value
    def __repr__(self):
        return '\n'.join('\n'.join([str(k)]+['\t'+x for x in str(v).split('\n')]) for k,v in self.dict.items())

class Estimator:
    resource = namedtuple(taiex = General('taiex'), otc = General('otc'))
    def __init__(self,params):
        assert hasattr(self,'call'), 'call not defined'
        self.config = namedtuple(
            params={k:v for k,v in params.items() if ('__' not in k) & (k != 'self')},
            structure={'call':self.call.__name__})
    def __call__(self):
        for_detector, for_trader = self.call(self.resource)
        return for_detector, for_trader
    @classmethod
    def build(clz,call=None):
        func_dict = {k:v for k,v in vars().items() if k!='clz' and v is not None}
        return type(f'Sub{clz.__name__}',(clz,),func_dict)

class Detector:
    def __init__(self,params):
        assert hasattr(self,'call'), 'call not defined'
        self.config = namedtuple(
            params={k:v for k,v in params.items() if ('__' not in k) & (k != 'self')},
            structure={'call':self.call.__name__})
    
    def __call__(self,stock,trend):
        resource = namedtuple(stock=stock,trend=trend)
        signal = self.call(resource)
        detection = signal[signal].index
        return detection
    
    @classmethod
    def build(clz,call=None):
        func_dict = {k:v for k,v in vars().items() if k!='clz' and v is not None}
        return type(f'Sub{clz.__name__}',(clz,),func_dict)
    
class Trader:
    def __init__(self,params):
        assert hasattr(self,'call'), 'call not defined'
        assert hasattr(self,'cancel'), 'cancel not defined'
        assert hasattr(self,'buy'), 'buy not defined'
        assert hasattr(self,'fail'), 'fail not defined'
        assert hasattr(self,'flush'), 'flush not defined'
        assert hasattr(self,'success'), 'success not defined'
        self.config = namedtuple(
            params={k:v for k,v in params.items() if ('__' not in k) & (k != 'self')},
            structure={
                'cancel':self.cancel.__name__,
                'buy':self.buy.__name__,
                'fail':self.fail.__name__,
                'flush':self.flush.__name__,
                'success':self.success.__name__})

    def __call__(self,detect_dates,stock,trend):
        cancel=[]
        buy=[]
        sell=[]
        complete=[]
        resource = namedtuple(stock=stock,trend=trend)
        self.call(resource)
        for detect_date in detect_dates:
            instruction,record = self.instruct(detect_date,resource)
            if record.state == 'end':
                data = dict(
                    ID=stock.info.ID,
                    name=stock.info.name,
                    industry=stock.info.industry,
                    market=stock.info.market,
                    detect_date=record.detect.date,
                    detect_price=record.detect.price,
                    buy_date=record.buy.date,
                    buy_price=record.buy.price,
                    buy_instruct=record.buy.instruct,
                    weight=record.buy.weight,
                    sell_date=record.sell.date,
                    sell_price=record.sell.price,
                    sell_reason=record.sell.reason,
                    sell_instruct=record.sell.instruct,
                    days=record.current.day,
                    note='_'.join(record.note),
                    current_date=stock.ohlcv.index[-1],
                    current_price=stock.ohlcv.close[-1])
                complete.append(data)
            else:
                if instruction.action == 'sell':
                    data = dict(
                        ID=stock.info.ID,
                        name=stock.info.name,
                        industry=stock.info.industry,
                        market=stock.info.market,
                        detect_date=record.detect.date,
                        detect_price=record.detect.price,
                        buy_date=record.buy.date,
                        buy_price=record.buy.price,
                        buy_instruct=record.buy.instruct,
                        weight=record.buy.weight,
                        sell_order=instruction.order,
                        sell_reason=instruction.reason,
                        sell_instruct=instruction.instruct,
                        days=record.current.day,
                        note='_'.join(record.note),
                        current_date=record.current.date,
                        current_price=record.current.ohlcv.close)
                    sell.append(data)
                else:
                    if instruction.action == 'buy':
                        data = dict(
                            ID=stock.info.ID,
                            name=stock.info.name,
                            industry=stock.info.industry,
                            market=stock.info.market,
                            detect_date=record.detect.date,
                            detect_price=record.detect.price,
                            buy_order=instruction.order,
                            buy_instruct=instruction.instruct,
                            weight=instruction.weight,
                            current_date=record.current.date,
                            current_price=record.current.ohlcv.close,
                            days=record.current.day)
                        buy.append(data)
                    else:
                        data = dict(
                            ID=stock.info.ID,
                            name=stock.info.name,
                            industry=stock.info.industry,
                            market=stock.info.market,
                            detect_date=record.detect.date,
                            detect_price=record.detect.price,
                            cancel_date=record.current.date,
                            cancel_price=record.current.ohlcv.open,
                            cancel_instruct=instruction.instruct,
                            days=record.current.day)
                        cancel.append(data)
        cancel = pd.DataFrame(cancel, dtype='object')
        buy = pd.DataFrame(buy, dtype='object')
        sell = pd.DataFrame(sell, dtype='object')
        complete = pd.DataFrame(complete, dtype='object')
        return cancel,buy,sell,complete

    def instruct(self,detect_date,resource):
        state=iter(['detect','buy','sell','end'])
        
        record = namedtuple(
            detect=namedtuple(date=pd.NaT, price=np.nan),
            buy = namedtuple(date=pd.NaT, price=np.nan, instruct='', weight=np.nan),
            sell = namedtuple(date=pd.NaT, price=np.nan, reason='', instruct=''),
            current = namedtuple(day=np.nan, date=pd.NaT, ohlcv=None),
            note=[],
            state=next(state))

        for day,(date,ohlcv) in enumerate(resource.stock.ohlcv.loc[detect_date:].iterrows()):
            
            record.current.day = day
            record.current.date = date
            record.current.ohlcv = ohlcv
            
            if record.state == 'sell':
                result = sell_match(instruction.order,ohlcv)
                if not np.isnan(result):
                    record.sell.date = date
                    record.sell.price = result
                    record.sell.reason = instruction.reason
                    record.sell.instruct = instruction.instruct
                    record.state = next(state)
                    break
            elif record.state == 'buy':
                if instruction.action == 'buy':
                    result = buy_match(instruction.order,ohlcv)
                    if not np.isnan(result):
                        record.buy.date = date
                        record.buy.price = result
                        record.buy.instruct = instruction.instruct
                        record.buy.weight = instruction.weight
                        record.state = next(state)
                else:
                    break
            else :
                record.detect.date = date
                record.detect.price = ohlcv.close
                record.state = next(state)
            
            if record.state == 'sell':
                instruction = self.__sell(resource,record)
            else:
                instruction = self.__trace(resource,record)
        
        return instruction,record
    
    def __parse_order(self,order):
        if order is None:
            instruct = ''
            weight = np.nan
        else:
            if isinstance(order,tuple):
                try:
                    order,instruct = order
                    weight = 1
                except ValueError:
                    order,instruct,weight = order
            else:
                weight = 1
                instruct = ''
            assert 0 < weight and weight <= 1
        return order,str(instruct),weight

    def __trace(self,resource,record):
        cancel_order,cancel_instruct = self.__cancel(resource,record)
        if cancel_order:
            return namedtuple(action='cancel',instruct=cancel_instruct)
        else:
            buy_order,buy_instruct,weight = self.__buy(resource,record)
            return namedtuple(action='buy',order=buy_order,instruct=buy_instruct,weight=weight)

    def __buy(self,resource,record):
        order = self.buy(resource,record)
        order,instruct,weight = self.__parse_order(order)
        order = approximate(order)
        return order,instruct,weight

    def __cancel(self,resource,record):
        order = self.cancel(resource,record)
        order,instruct,_ = self.__parse_order(order)
        if order is None:
            order = False
        assert isinstance(order,bool)
        return order,instruct
    
    def __fail(self,resource,record):
        order = self.fail(resource,record)
        order,instruct,_ = self.__parse_order(order)
        order = approximate(order)
        if isinstance(order,(int,float)) and (order is not True):
            if record.current.ohlcv.close <= order:
                order = True
            else:
                order = False
        else:
            if np.isnan(order):
                order = False
        return order,instruct
    
    def __flush(self,resource,record):
        order = self.flush(resource,record)
        order,instruct,_ = self.__parse_order(order)
        order = approximate(order)
        return order,instruct
        
    def __success(self,resource,record):
        order = self.success(resource,record)
        order,instruct,_ = self.__parse_order(order)
        order = approximate(order)
        return order,instruct
    
    def __sell(self,resource,record):
        fail_order,fail_instruct = self.__fail(resource,record)
        if fail_order:
            order = True
            reason = 'fail'
            instruct = fail_instruct
        else:
            flush_order,flush_instruct = self.__flush(resource,record)
            success_order,success_instruct = self.__success(resource,record)
            if (flush_order is True) or (success_order is True):
                order = True
                reason=' & '.join([x[0] for x in zip(['flush','success'],[flush_order,success_order]) if x[1] is True])
                instruct=' & '.join([x[0] for x in zip([flush_instruct,success_instruct],[flush_order,success_order]) if x[1] is True])
            elif not np.isnan(flush_order) and not np.isnan(success_order):
                if success_order > flush_order:
                    order=flush_order
                    reason='flush'
                    instruct=flush_instruct
                elif flush_order > success_order:
                    order=success_order
                    reason='success'
                    instruct=success_instruct
                else:
                    order=success_order
                    reason='flush & success'
                    instruct=' & '.join([flush_instruct,success_instruct])
            elif not np.isnan(success_order):
                order=success_order
                reason='success'
                instruct=success_instruct
            elif not np.isnan(flush_order):
                order=flush_order
                reason='flush'
                instruct=flush_instruct
            else:
                order=np.nan
                reason='nan'
                instruct=''
        return namedtuple(action='sell',order=order,reason=reason,instruct=instruct)
    
    @classmethod
    def build(clz,cancel=None,buy=None,fail=None,flush=None,success=None):
        func_dict = {k:v for k,v in vars().items() if k!='clz' and v is not None}
        return type(f'Sub{clz.__name__}',(clz,),func_dict)
    
def ceil(price):
    if price < 50:
        if price < 10:
            return math.ceil(price*100)/100
        else:
            return math.ceil(price*20)/20
    else:
        if price < 100:
            return math.ceil(price*10)/10
        else:
            if price < 500:
                return math.ceil(price*2)/2
            else:
                if price < 1000:
                    return math.ceil(price)
                else:
                    return math.ceil(price/5)*5

trans = {None: lambda price: np.nan,
         np.nan: lambda price: np.nan,
         False: lambda price: np.nan,
         True: lambda price: True}

approximate = lambda price: trans.get(price,ceil)(price)

def buy_match(order,ohlcv):
    if order is True:
        return ohlcv.open
    elif isinstance(order,(int,float)):
        if ohlcv.open <= order:
            return ohlcv.open
        elif ohlcv.low <= order:
            return order
    return np.nan
    
def sell_match(order,ohlcv):
    if order is True:
        return ohlcv.open
    elif isinstance(order,(int,float)):
        if ohlcv.open >= order:
            return ohlcv.open
        elif ohlcv.high >= order:
            return order
    return np.nan

class Adjuster:
    def __init__(self,params):
        assert hasattr(self,'rate'), 'rate not defined'

        self.valid_period = datetime.timedelta(days=params['valid_period'])
        self.groupby_key = params.setdefault('groupby_key',None)
        self.start_by = params.setdefault('start_by','detect_date')
        self.end_by = params.setdefault('end_by','sell_date')

        self.config = namedtuple(
            params={k:v for k,v in params.items() if ('__' not in k) & (k != 'self')},
            structure={'rate':self.rate.__name__})
        
    def __call__(self,result):
        self.adjust(result.complete,result.buy)
        self.adjust(result.complete,result.sell)
        self.adjust(result.complete,result.complete)

    def __rate(self,keys,ref):
        output = self.rate(keys,ref)
        if output is None:
            output = 1
        return output
        
    def adjust(self,past,df):
        if not df.empty:
            if self.valid_period.days == 0:
                df.loc[:,'adjust'] = 1
            else:
                unique_dates = df.detect_date.drop_duplicates().sort_values()
                for end_date in unique_dates:
                    start_date = end_date - self.valid_period
                    valid_past = past.loc[(past[self.start_by]>start_date)&(past[self.end_by]<=end_date)]
                    if self.groupby_key is not None:
                        valid_past_groups = valid_past.groupby(self.groupby_key).groups
                        for keys,group_index in df.loc[df.detect_date==end_date].groupby([self.groupby_keys]).groups.iteritems():
                            ref = valid_past.loc[valid_past_groups.get(keys,[])]
                            if ref.empty:
                                df.loc[group_index,'adjust'] = 1
                            else:
                                keys = {k:v for k,v in zip(self.groupby_key,keys)}
                                df.loc[group_index,'adjust'] = self.__rate(keys,ref)
                    else:
                        if valid_past.empty:
                            df.loc[df.detect_date==end_date,'adjust'] = 1
                        else:
                            df.loc[df.detect_date==end_date,'adjust'] = self.__rate({},valid_past)
            df.loc[:,'adjusted_weight'] = df['weight'] * df['adjust']
            
    @classmethod
    def build(clz,call=None):
        func_dict = {k:v for k,v in vars().items() if k!='clz' and v is not None}
        return type(f'Sub{clz.__name__}',(clz,),func_dict)
    
class Model:
    def __init__(self,name=None):
        self.name = name
        
    def compile(self,estimator,detector,trader,adjuster):
        self.estimator = estimator
        self.detector = detector
        self.trader = trader
        self.adjuster = adjuster
        
        assert hasattr(estimator,'config'),'Please ADD: super().__init__(vars()) in the first line of estimator.__init__()'
        assert hasattr(detector,'config'),'Please ADD: super().__init__(vars()) in the first line of detector.__init__()'
        assert hasattr(trader,'config'),'Please ADD: super().__init__(vars()) in the first line of trader.__init__()'
        assert hasattr(adjuster,'config'),'Please ADD: super().__init__(valid_period,vars()) in the first line of adjuster.__init__(valid_period,groupby_key=None,start_by="detect_date",end_by="sell_date",**params)'

        self.config = namedtuple(
            model_name=self.name,
            estimator=estimator.config,
            detector=detector.config,
            trader=trader.config,
            adjuster=adjuster.config)
    
    @property
    def config_table(self):
        def decompose(config,which):
            component = getattr(config,which)
            param = pd.DataFrame().append(component.params,ignore_index=True).add_prefix(which[0].upper()+'_P_')
            structure = pd.DataFrame().append(component.structure,ignore_index=True).add_prefix(which[0].upper()+'_F_')
            return pd.concat([structure,param],axis=1)
        def model_description(config):
            E = decompose(config,'estimator')
            D = decompose(config,'detector')
            T = decompose(config,'trader')
            A = decompose(config,'adjuster')
            return pd.concat([E,D,T,A],axis=1)
        return model_description(self.config)
        

    def create_result_container(self):
        cancel = pd.DataFrame(
            columns=[
                'ID',
                'name',
                'industry',
                'market',
                'detect_date',
                'detect_price',
                'cancel_date',
                'cancel_price',
                'cancel_instruct',
                'days'])
        buy = pd.DataFrame(
            columns=[
                'ID',
                'name',
                'industry',
                'market',
                'detect_date',
                'detect_price',
                'buy_order',
                'buy_instruct',
                'weight',
                'adjust',
                'adjusted_weight',
                'current_date',
                'current_price',
                'days'])
        sell = pd.DataFrame(
            columns=[
                'ID',
                'name',
                'industry',
                'market',
                'detect_date',
                'detect_price',
                'buy_date',
                'buy_price',
                'buy_instruct',
                'weight',
                'adjust',
                'adjusted_weight',
                'sell_order',
                'sell_reason',
                'sell_instruct',
                'days',
                'note',
                'current_date',
                'current_price',
                'floating_return',
                'floating_revenue',
                'floating_return_after_tax',
                'floating_revenue_after_tax'])
        complete = pd.DataFrame(
            columns=[
                'ID',
                'name',
                'industry',
                'market',
                'detect_date',
                'detect_price',
                'buy_date',
                'buy_price',
                'buy_instruct',
                'weight',
                'adjust',
                'adjusted_weight',
                'sell_date',
                'sell_price',
                'sell_reason',
                'sell_instruct',
                'days',
                'note',
                'current_date',
                'current_price',
                'floating_return',
                'floating_revenue',
                'floating_return_after_tax',
                'floating_revenue_after_tax',
                'realized_return',
                'realized_revenue',
                'realized_return_after_tax',
                'realized_revenue_after_tax'])
        result=namedtuple(cancel=cancel,buy=buy,sell=sell,complete=complete)
        return result

    def advice(self,*transections):
        result = self.create_result_container()
        trend_for_detect,trend_for_trader = self.estimator()
        for transaction in transections:
            detect_dates = self.detector(transaction.stock,trend_for_detect)
            assert transaction.detect_date in detect_dates
            cancel,buy,sell,complete = self.trader([transaction.detect_date],transaction.stock,trend_for_trader)
            result.cancel = result.cancel.append(cancel,ignore_index=True)
            result.buy = result.buy.append(buy,ignore_index=True)
            result.sell = result.sell.append(sell,ignore_index=True)
            result.complete = result.complete.append(complete,ignore_index=True)
            
    def calculate_return(self,table,return_col,start_col,end_col):
        table.loc[:,return_col] = table[end_col] / table[start_col]
        
    def calculate_return_after_tax(self,table,return_col,start_col,end_col):
        table.loc[:,return_col] = (table[end_col]*(1-transactions_tax-processing_fee)/table[start_col])-processing_fee
        
    def calculate_revenue(self,table,revenue_col,return_col):
        table.loc[table[return_col]>1,revenue_col] = 'gain'
        table.loc[table[return_col]<1,revenue_col] = 'loss'
        table.loc[table[return_col]==1,revenue_col] = 'tie'


    def fit(self,dataset,save_folder=None):
        result = self.create_result_container()
        trend_for_detect,trend_for_trader = self.estimator()
        for stock in dataset:
            detect_dates = self.detector(stock,trend_for_detect)
            cancel,buy,sell,complete = self.trader(detect_dates,stock,trend_for_trader)
            result.cancel = result.cancel.append(cancel,ignore_index=True)
            result.buy = result.buy.append(buy,ignore_index=True)
            result.sell = result.sell.append(sell,ignore_index=True)
            result.complete = result.complete.append(complete,ignore_index=True)
            
        self.calculate_return(result.sell,'floating_return','buy_price','current_price')
        self.calculate_revenue(result.sell,'floating_revenue','floating_return')
        self.calculate_return_after_tax(result.sell,'floating_return_after_tax','buy_price','current_price')
        self.calculate_revenue(result.sell,'floating_revenue_after_tax','floating_return_after_tax')
        
        self.calculate_return(result.complete,'floating_return','buy_price','current_price')
        self.calculate_revenue(result.complete,'floating_revenue','floating_return')
        self.calculate_return_after_tax(result.complete,'floating_return_after_tax','buy_price','current_price')
        self.calculate_revenue(result.complete,'floating_revenue_after_tax','floating_return_after_tax')
        self.calculate_return(result.complete,'realized_return','buy_price','sell_price')
        self.calculate_revenue(result.complete,'realized_revenue','realized_return')
        self.calculate_return_after_tax(result.complete,'realized_return_after_tax','buy_price','sell_price')
        self.calculate_revenue(result.complete,'realized_revenue_after_tax','realized_return_after_tax')
        
        
        result.cancel = result.cancel.convert_dtypes(convert_integer=False)
        result.buy = result.buy.convert_dtypes(convert_integer=False)
        result.sell = result.sell.convert_dtypes(convert_integer=False)
        result.complete = result.complete.convert_dtypes(convert_integer=False)
        
        self.adjuster(result)
        self.result = result
        return result

    def save(self,save_path='',drive='D'):
        save_path = pathlib.Path(save_path,self.name)
        if drive == 'D':
            save_path = pathlib.Path('D:\\','StockLab').joinpath(save_path)
        save_path.mkdir(parents=True)

        config_text = save_path.joinpath('config.txt')
        config_pkl = save_path.joinpath('config.pkl')
        result_file = save_path.joinpath('result.pkl')
        summary_file = save_path.joinpath('summary.pkl')

        config_text.write_text(str(self.config))
        config_pkl.write_bytes(pickle.dumps(self.config_table))
        result_file.write_bytes(pickle.dumps(self.result.dict))
        summary_file.write_bytes(pickle.dumps(self.summary()))
        
    def summary(self):
        table = self.result.complete
        
        output = pd.DataFrame(index=[0])
        output['model'] = self.name
        output['count'] = table.shape[0]
        output['mean'] = table['realized_return_after_tax'].mean()
        output['weighted_mean'] = (table['realized_return_after_tax']*table['weight']).sum()/table['weight'].sum()
        output['adjusted_mean'] = (table['realized_return_after_tax']*table['adjusted_weight']).sum()/table['adjusted_weight'].sum()
        output['geo_mean'] = 2**np.log2(table['realized_return_after_tax']).mean()
        output['geo_std'] = 2**np.log2(table['realized_return_after_tax']).std()
        output['gain'] = (table['realized_revenue_after_tax']=='gain').mean()
        output['tie'] = (table['realized_revenue_after_tax']=='tie').mean()
        output['loss'] = (table['realized_revenue_after_tax']=='loss').mean()
        output['success'] = (table['sell_reason']=='success').mean()
        output['flush'] = (table['sell_reason']=='flush').mean()
        output['fail'] = (table['sell_reason']=='fail').mean()
        
        config = self.config_table
        output = pd.concat([output,config],axis=1)
        return output
        
class GridSearch:
    def __init__(self,project_name,add_mode=False):
        self.project_name = pathlib.Path(project_name)
        self.project_name.mkdir(parents=True,exist_ok=add_mode)
        self.exist_model = {k.joinpath('config.txt').read_text()[18:]:k.joinpath('summary.pkl') for k in self.project_name.iterdir()}
        
        self.iter_counts = 1
        self.add_mode = add_mode
        self.EstimatorABC = namedtuple(trunk=None, func_combs=None, params=None)
        self.DetectorABC = namedtuple(trunk=None, func_combs=None, params=None)
        self.TraderABC = namedtuple(trunk=None, func_combs=None, params=None)
        self.AdjusterABC = namedtuple(trunk=None, func_combs=None, params=None)
        
        self.compile_dict = dict(estimator=self.EstimatorABC,
                                 detector=self.DetectorABC,
                                 trader=self.TraderABC,
                                 adjuster=self.AdjusterABC)
        
    def grid(self,*params):
        self.compile_dict[self.compile_object].params = itertools.product(*params)
        del self.compile_object
        self.iter_counts *= len(list(itertools.product(*params)))
        
    def pair(self,params):
        self.compile_dict[self.compile_object].params = itertools.product(params)
        del self.compile_object
        self.iter_counts *= len(list(itertools.product(params)))
        
    def compile_estimator(self, trunk, call=[None]):
        self.compile_object = 'estimator'
        self.EstimatorABC.trunk = trunk
        self.EstimatorABC.func_combs = itertools.product(call)
        self.iter_counts *= len(call)
        return self
        
    def compile_detector(self, trunk, call=[None]):
        self.compile_object = 'detector'
        self.DetectorABC.trunk = trunk
        self.DetectorABC.func_combs = itertools.product(call)
        self.iter_counts *= len(call)
        return self

    def compile_trader(self, trunk, cancel=[None], buy=[None], fail=[None], flush=[None], success=[None]):
        self.compile_object = 'trader'
        self.TraderABC.trunk = trunk
        self.TraderABC.func_combs = itertools.product(cancel,buy,fail,flush,success)
        self.iter_counts *= len(cancel)*len(buy)*len(fail)*len(flush)*len(success)
        return self
        
    def compile_adjuster(self,trunk,rate=[None]):
        self.compile_object = 'adjuster'
        self.AdjusterABC.trunk = trunk
        self.AdjusterABC.func_combs = itertools.product(rate)
        self.iter_counts *= len(rate)
        return self

    def fit(self,data,save_drive='D'):
        table = pd.DataFrame()
        structure_combs = itertools.product(self.EstimatorABC.func_combs,
                                            self.DetectorABC.func_combs,
                                            self.TraderABC.func_combs,
                                            self.AdjusterABC.func_combs)
        hyperparams = itertools.product(self.EstimatorABC.params,
                                        self.DetectorABC.params,
                                        self.TraderABC.params,
                                        self.AdjusterABC.params)
        output = widgets.Output()
        display(output)
        start_time = datetime.datetime.now()
        
        i = 1
        n = len(list(self.project_name.iterdir()))
        for estimator_funcs,detector_funcs,trader_funcs,adjuster_funcs in structure_combs:
            Estimator = self.EstimatorABC.trunk.build(*estimator_funcs)
            Detector = self.DetectorABC.trunk.build(*detector_funcs)
            Trader = self.TraderABC.trunk.build(*trader_funcs)
            Adjuster = self.AdjusterABC.trunk.build(*adjuster_funcs)
            for estimator_param, detector_param, trader_param, adjuster_param in hyperparams:
                estimator = Estimator(*estimator_param)
                detector = Detector(*detector_param)
                trader = Trader(*trader_param)
                adjuster = Adjuster(*adjuster_param)
                model = Model(f'{n:0>5}')
                model.compile(estimator,detector,trader,adjuster)
                try:
                    exist_model_path = self.exist_model[str(model.config)[18:]]
                    table = table.append(pickle.loads(exist_model_path.read_bytes()),ignore_index=True,sort=False)
                except KeyError:
                    with output:
                        clear_output(wait=True)
                        print(f'Runtime: {datetime.datetime.now() - start_time}  {i}/{self.iter_counts}')
                        print(str(model.config))
                    model.fit(data)
                    model.save(self.project_name,save_drive)
                    table = table.append(model.summary(),ignore_index=True,sort=False)
                    n += 1
                i += 1
        return table

    # def summary(self,table,config):
    #     output = OrderedDict()

    #     output['count'] = table.shape[0]

    #     def return_(table,column):
    #         weight = table['weight']
    #         adjusted = table['adjusted_weight']
    #         col = table[column]
    #         mean = col.mean()
    #         weighted_mean = (col*weight).sum()/weight.sum()
    #         adjusted_mean = (col*adjusted).sum()/adjusted.sum()
    #         geo_mean = 2**np.log2(col).mean()
    #         geo_std = 2**np.log2(col).std()
    #         return {
    #             f'{column}_mean':mean,
    #             f'{column}_weighted_mean':weighted_mean,
    #             f'{column}_adjusted_mean':adjusted_mean,
    #             f'{column}_geo_mean':geo_mean,
    #             f'{column}_geo_std':geo_std}

    #     output.update(return_(table,'realized_return'))
    #     output.update(return_(table,'realized_return_after_tax'))
        
    #     def revenue_(table,column):
    #         col = table[column]
    #         gain = (col=='gain').mean()
    #         tie = (col=='tie').mean()
    #         loss = (col=='loss').mean()
    #         return {
    #             f'{column}_gain':gain,
    #             f'{column}_tie':tie,
    #             f'{column}_loss':loss
    #         }

    #     output.update(revenue_(table,'realized_revenue'))
    #     output.update(revenue_(table,'realized_revenue_after_tax'))
        
    #     output['success'] = (table['sell_reason']=='success').mean()
    #     output['flush'] = (table['sell_reason']=='flush').mean()
    #     output['fail'] = (table['sell_reason']=='fail').mean()
        
    #     output['config'] = str(config)
    #     output['model'] = config.model_name
    #     return output
    
    
    

class Transaction:
    def __init__(self,id_or_name,detect_date,buy_date,buy_price,volume):
        pass
