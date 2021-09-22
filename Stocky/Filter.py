import pandas as pd
import numpy as np

class NonContinuous:
    def __init__(self):
        pass
    def __call__(self,data):
        return (data != data.shift(1).fillna(False)) & data

class Switch:
    def __init__(self):
        pass
    def __call__(self,data):
        return data != data.shift(1).fillna(False)

class Refractory:
    def __init__(self,period):
        self.period = period
        def filter_func(data,period):
            keep = []
            for i in np.where(data.values)[0]:
                if not keep:
                    keep.append(i)
                else:
                    if i - keep[-1] > period:
                        keep.append(i)
            ser = pd.Series(False,index=data.index)
            ser[keep] = True
            return ser
        self.filter_func = filter_func
    def __call__(self,data):
        if isinstance(data,pd.Series):
            return self.filter_func(data,self.period)
        else:    
            return pd.DataFrame({col:self.filter_func(data[col],self.period) for col in data.columns})
