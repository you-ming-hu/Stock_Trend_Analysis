import pathlib
import pandas as pd
from IPython.display import clear_output
import functools
import datetime
from collections import namedtuple
import ipywidgets as widgets


Overview = pd.read_csv(
    pathlib.Path('Stocky','Database','depository','overview','data.csv').as_posix(),
    index_col=0,
    dtype={'ID':str},
    encoding='utf8')
    
def ActiveStocks():
    path = pathlib.Path('Stocky','Database','agent','individual','technical','price_volume','active_stocks.csv')
    IDs = pd.read_csv(path.as_posix(),encoding='utf8',index_col=0,dtype=str).squeeze().tolist()
    return StockPile(IDs)

def SelectIDs(IDs):
    return StockPile([str(x) for x in IDs])

def SelectNames(names):
    IDs = Overview.set_index('name').loc[names].ID.tolist()
    return StockPile(IDs)

def Range(*s):
    IDs = list(x.name for x in pathlib.Path('Stocky','Database','depository','individual').iterdir())[slice(*s)]
    return StockPile(IDs)

def All():
    IDs = list(x.name for x in pathlib.Path('Stocky','Database','depository','individual').iterdir())
    return StockPile(IDs)
    
class StockPile:
    def __init__(self,ID_list):
        self.table = Overview.set_index('ID').loc[ID_list]
        pass
    def random(self,portion):
        pass
    def __iter__(self):
        if not hasattr(self,'output'):
            self.output = widgets.Output()
            display(self.output)
        
        start_time = datetime.datetime.now()
        for i,ID in enumerate(self.table.index):
            stock = Stock(ID)
            with self.output:
                clear_output(wait=True)
                print(f'{i+1}/{self.table.shape[0]} {stock.info}')
                print(f'Runtime: {datetime.datetime.now() - start_time}')
            yield stock

info_container = namedtuple('StockInfo',['ID','name','industry','market'])
class Stock:
    def __init__(self,ID_or_name):
        ID_or_name = str(ID_or_name)
        try:
            info = Overview.set_index('ID').loc[ID_or_name]
            stockID = ID_or_name
            name  = info['name']
        except KeyError:
            info = Overview.set_index('name').loc[ID_or_name]
            stockID = info['ID']
            name  = ID_or_name
        finally:
            self.__path = pathlib.Path('Stocky','Database','depository','individual',stockID)
            self.info = info_container(ID=stockID, name=name, industry=info['industry'], market=info['market'])

    def __repr__(self):
        options = list(x.as_posix() for x in self.__path.rglob('data.csv'))
        return '\n'.join([str(self.info)]+options)

    @functools.cached_property
    def ohlcv(self):
        table = pd.read_csv(self.__path.joinpath('technical','price_volume','data.csv').as_posix(),
                            encoding='utf8',
                            index_col='date',
                            parse_dates=True,
                            infer_datetime_format='%Y-%m-%d',
                            dtype={'ID':str})
        table = table[['open','high','low','close','volume','count','capital']]
        table = table.loc[(table[['open','high','low','close']] != 0).all(axis=1)]
        return table
    
class General:
    def __init__(self,which):
        self.__path = pathlib.Path('Stocky','Database','depository','general',which)

    @functools.cached_property
    def ohlcv(self):
        table = pd.read_csv(self.__path.joinpath('technical','price_volume','data.csv').as_posix(),
                            encoding='utf8',
                            index_col='date',
                            parse_dates=True,
                            infer_datetime_format='%Y-%m-%d')
        return table
    

