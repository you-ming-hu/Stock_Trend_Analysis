import numpy as np
import pandas as pd

class Increasing:
    def __init__(self,restrict=True):
        self.restrict = restrict
    def __call__(self,data):
        assert isinstance(data,pd.DataFrame)
        if self.restrict:
            return (data < data.shift(-1,axis=1)).iloc[:,:-1].all(axis=1)
        else:
            return (data <= data.shift(-1,axis=1)).iloc[:,:-1].all(axis=1)

class Decreasing:
    def __init__(self,restrict=True):
        self.restrict = restrict
    def __call__(self,data):
        assert isinstance(data,pd.DataFrame)
        if self.restrict:
            return (data > data.shift(-1,axis=1)).iloc[:,:-1].all(axis=1)
        else:
            return (data >= data.shift(-1,axis=1)).iloc[:,:-1].all(axis=1)

# class Ascending:
#     def __init__(self,window,restrict=True):
#         self.window = window
#         self.restrict = restrict
#     def __call__(self,data):
#         if self.restrict:
#             return (data > data.shift(1)).rolling(self.window).apply(np.all) == 1
#         else:
#             return (data >= data.shift(1)).rolling(self.window).apply(np.all) == 1

# class Descending:
#     def __init__(self,window,restrict=True):
#         self.window = window
#         self.restrict = restrict
#     def __call__(self,data):
#         if self.restrict:
#             return (data < data.shift(1)).rolling(self.window).apply(np.all) == 1
#         else:
#             return (data <= data.shift(1)).rolling(self.window).apply(np.all) == 1

class Ascending:
    def __init__(self,restrict=True):
        self.restrict = restrict
    def __call__(self,data):
        if self.restrict:
            return data > data.shift(1)
        else:
            return data >= data.shift(1)

class Descending:
    def __init__(self,restrict=True):
        self.restrict = restrict
    def __call__(self,data):
        if self.restrict:
            return data < data.shift(1)
        else:
            return data <= data.shift(1)