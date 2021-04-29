# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:44:35 2021

@author: zaju9001
"""
import pandas as pd
pd.set_option("display.max_columns",100)
import warnings
warnings.filterwarnings("ignore")
import re
import unicodedata

from plotly.offline import plot,iplot
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
import plotly.express as px#graficos express
import haversine as hs

import cufflinks as cf
cf.go_offline()
import numpy as np
import nltk

import matplotlib.pyplot as plt
import seaborn as sns


from pyecharts.charts import Pie
from pyecharts import options as opts

from sklearn.model_selection import train_test_split

from matplotlib import pyplot

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

class drop_columns(TransformerMixin):
    def fit(self, X, y=None):
        self.columns = ["weekday", "dropoff_latitude","pickup_latitude", "pickup_datetime",
                        "pickup_longitude", "dropoff_longitude", "key", "fare_amount", "fare_class"]
        return self

    def transform(self, X):
        X.drop(columns = self.columns, inplace = True )
        return X

class add_cols(TransformerMixin):
    def fit(self, X, y=None):
        #cosas que aprende
        return self

    def transform(self, X):
        X['pickup_datetime'] =  pd.to_datetime(X['pickup_datetime'], infer_datetime_format=True)
        
        X["dist_haversine"] = X.apply(lambda x:hs.haversine((x["pickup_latitude"],x["pickup_longitude"]),
                                                 (x["dropoff_latitude"],x["dropoff_longitude"])), axis = 1)
    
        X["month"] = X["pickup_datetime"].map(lambda x: x.month)
        X["hour"] = X["pickup_datetime"].map(lambda x: x.hour)
        X["weekday"] = X["pickup_datetime"].map(lambda x: x.weekday())
        X["weekend"] = X["weekday"].apply(lambda x: 1 if x >= 5 else 0)
        

        return X


