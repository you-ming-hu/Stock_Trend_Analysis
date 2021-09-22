import numpy as np
import pandas as pd

class DownSampleBase:
    def __init__(self,freq):
        assert isinstance(freq,str)
        self.freq = freq

class Last(DownSampleBase):
    def __call__(self,data):
        data = data.reset_index()
        return data.resample(self.freq,on='date').last().set_index('date').squeeze(axis=1).dropna()

class Max(DownSampleBase):
    def __call__(self,data):
        data = data.reset_index()
        apply_rule = {c:{'date':'last'}.get(c,'max') for c in data.coulmns}
        return data.resample(self.freq,on='date').apply(apply_rule).set_index('date').squeeze(axis=1).dropna()
    
class OHLCV(DownSampleBase):
    default = {'open':'first','low':'min','high':'max','close':'last','volume':'sum','count':'sum','capital':'sum','date':'last'}
    def __call__(self,data):
        data = data.reset_index()
        apply_rule = {x:self.default[x] for x in data.columns}
        return data.resample(self.freq,on='date').apply(apply_rule).set_index('date').squeeze(axis=1).dropna()

class CummulativeMax(DownSampleBase):
    def __call__(self,data):
        return data.resample(self.freq).apply(lambda df:df.cummax())
    
class CummulativeMin(DownSampleBase):
    def __call__(self,data):
        return data.resample(self.freq).apply(lambda df:df.cummin())
    
class CummulativeOHLCV(DownSampleBase):
    default = {'open': lambda ser: ser[0],
               'high': lambda ser: ser.cummax(),
               'low': lambda ser: ser.cummin(),
               'close': lambda ser: ser,
               'volume': lambda ser: ser.cumsum(),
               'count': lambda ser: ser.cumsum(),
               'capital': lambda ser: ser.cumsum()}
    def __call__(self,data):
        apply_func = lambda df: pd.DataFrame({c:self.default[c](df[c]) for c in df.columns})
        return data.resample(self.freq).apply(apply_func)

class Padding:
    def __init__(self,fillna):
        assert isinstance(fillna,bool)
        self.fillna = fillna
        
    def __call__(self,template,data):
        if self.fillna:
            return data.reindex(template.index,method='ffill')
        else:
            return data.reindex(template.index)
            
class Align:
    def __init__(self,fillna):
        assert isinstance(fillna,bool)
        self.fillna = fillna
        
    def __call__(self,template,data):
        data = data.reset_index().set_index('date',drop=False)
        data = data.asfreq('d',method='ffill').reindex(template.index)
        if not self.fillna:
            data.loc[data['date'] == data['date'].shift(1)] = np.nan
        data = data.drop('date',axis=1).squeeze(axis=1)
        return data