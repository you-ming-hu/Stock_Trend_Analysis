import pandas as pd
import numpy as np

from . import Resample
from . import Indicator

class MA:
    def __init__(self,freq,windows,**kwd):
        self.padding = Resample.Padding(True)
        self.resampler = Resample.Last(freq)
        if isinstance(windows,list):
            windows = np.array(windows)
        self.windows = windows
        self.sum = Indicator.RollingSum(windows-1,**kwd)
    
    def __call__(self,data):
        resample_data = self.resampler(data)
        rolling_sum = self.sum(resample_data)
        rolling_sum = self.padding(data,rolling_sum)
        rolling_sum = rolling_sum.shift(1)
        if isinstance(rolling_sum,pd.Series):
            rolling_sum = rolling_sum.to_frame()
        rolling_sum = rolling_sum + data.to_frame().values
        return (rolling_sum/self.windows).squeeze()

class RollingMax:
    def __init__(self,freq,windows,**kwd):
        self.padding = Resample.Padding(True)
        self.resampler = Resample.Last(freq)
        if isinstance(windows,list):
            windows = np.array(windows)
        self.windows = windows
        self.rolling_max = Indicator.RollingMax(windows-1,**kwd)
    def __call__(self,data):
        resample_data = self.resampler(data)
        rolling_max = self.rolling_max(resample_data)
        rolling_max = self.padding(data,rolling_max)
        rolling_max = rolling_max.shift(1)
        if isinstance(rolling_max,pd.Series):
            rolling_max = rolling_max.to_frame()
        return np.fmax(rolling_max,data.to_frame().values).squeeze()
    
class RollingMin:
    def __init__(self,freq,windows,**kwd):
        self.padding = Resample.Padding(True)
        self.resampler = Resample.Last(freq)
        if isinstance(windows,list):
            windows = np.array(windows)
        self.windows = windows
        self.rolling_min = Indicator.RollingMin(windows-1,**kwd)
    def __call__(self,data):
        resample_data = self.resampler(data)
        rolling_min = self.rolling_min(resample_data)
        rolling_min = self.padding(data,rolling_min)
        rolling_min = rolling_min.shift(1)
        if isinstance(rolling_min,pd.Series):
            rolling_min = rolling_min.to_frame()
        return np.fmin(rolling_min,data.to_frame().values).squeeze()

class EWMA:
    def __init__(self,freq,spans,**kwd):
        self.padding = Resample.Padding(True)
        self.resampler = Resample.Last(freq)
        if isinstance(spans,list):
            spans = np.array(spans)
        self.alpha = 1/(1+spans)
        self.ewma = Indicator.EWMA(spans,**kwd)

    def __call__(self,data):
        resample_data = self.resampler(data)
        ewma = self.ewma(resample_data)
        ewma = self.padding(data,ewma)
        ewma = ewma.shift(1)
        if isinstance(ewma,pd.Series):
            ewma = ewma.to_frame()
        return (data.to_frame().values * self.alpha + ewma * (1-self.alpha)).squeeze()
        

class MACD:
    def __init__(self,DIF_short,DIF_long,DEM_day):
        self.EWMA_short = EWMA(DIF_short)
        self.EWMA_long = EWMA(DIF_long)
        self.EWMA_DEM = EWMA(DEM_day)

    def __call__(self,data):
        DIF = self.EWMA_short(data) - self.EWMA_long(data)
        DEM = self.EWMA_DEM(DIF)
        OSC = DIF - DEM
        return pd.DataFrame(dict(DIF = DIF, DEM = DEM, OSC = OSC))

class RSV:
    def __init__(self,windows):
        self.min = RollingMin(windows,min_periods=1)
        self.max = RollingMax(windows,min_periods=1)
    def __call__(self,data):
        rolling_min = self.min(data)
        rolling_max = self.max(data)
        return (data - rolling_min) / (rolling_max - rolling_min) * 100

class KDJ:
    def __init__(self,rsv_window,rsvt,kt,init_value=50):
        self.rsv = RSV(rsv_window)
        self.init_value = init_value
        self.rsvt = rsvt
        self.kt = kt
    def __call__(self,data):
        rsv = self.rsv(data)

        k = np.zeros(rsv.shape)
        k[0] = self.init_value
        for i in range(1, k.shape[0]):
            k[i] = (k[i-1] * (self.rsvt - 1) + rsv[i]) / self.rsvt
        K = pd.Series(index = data.index, data = k)

        d = np.zeros(rsv.shape)
        d[0] = self.init_value
        for i in range(1, d.shape[0]):
            d[i] = (d[i-1] * (self.kt - 1) + k[i]) / self.kt
        D = pd.Series(index = data.index, data = d)

        J = 3*D - 2*K
        
        return pd.DataFrame({'K':K,'D':D,'J':J})
        
