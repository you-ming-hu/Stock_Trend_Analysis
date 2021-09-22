import pandas as pd
import numpy as np

class RollingBase:
    def __init__(self,func,windows,**kwd):
        self.func = func
        self.windows = windows
        self.kwd = kwd
    def __call__(self,data):
        assert isinstance(data,pd.Series)
        if isinstance(self.windows,int):
            return self.func(data.rolling(self.windows,**self.kwd))
        else:
            return  pd.DataFrame({i: self.func(data.rolling(w,**self.kwd)) for i,w in enumerate(self.windows)})

class Rolling(RollingBase):
    def __init__(self,func,windows,**kwd):
        def wraper(rolling_obj):
            return pd.core.window.rolling.Rolling.apply(rolling_obj,func)
        super().__init__(wraper,windows,**kwd)
    
class MA(RollingBase):
    def __init__(self,windows,**kwd):
        super().__init__(pd.core.window.rolling.Rolling.mean,windows,**kwd)

class RollingSum(RollingBase):
    def __init__(self,windows,**kwd):
        super().__init__(pd.core.window.rolling.Rolling.sum,windows,**kwd)

class RollingMax(RollingBase):
    def __init__(self,windows,**kwd):
        super().__init__(pd.core.window.rolling.Rolling.max,windows,**kwd)
        
class RollingMin(RollingBase):
    def __init__(self,windows,**kwd):
        super().__init__(pd.core.window.rolling.Rolling.min,windows,**kwd)

class EWM:
    def __init__(self,func,span,**kwd):
        self.func = func
        self.span = span
        self.kwd = kwd
    def __call__(self,data):
        assert isinstance(data,pd.Series)
        if isinstance(self.span,int):
            return self.func(data.ewm(self.span,adjust=False,**self.kwd))
        else:
            return  pd.DataFrame({i: self.func(data.ewm(s,adjust=False,**self.kwd)) for i,s in enumerate(self.span)})

class EWMA(EWM):
    def __init__(self,span,**kwd):
        super().__init__(pd.core.window.ewm.ExponentialMovingWindow.mean,span,**kwd)

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