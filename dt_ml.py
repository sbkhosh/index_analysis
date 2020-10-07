#!/usr/bin/python3

import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader as pdr
import requests_cache
import yaml

from datetime import datetime, timedelta
from dt_help import Helper
from sqlalchemy import create_engine

class MLPred():
    def __init__(self,input_directory, output_directory, input_prm_file, data):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file
        self.data = data
        
    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))
        
    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.forecast_out = self.conf.get('forecast_out')
        self.to_blacklist = self.conf.get('to_blacklist')
        
    @Helper.timing
    def ml_model(self):
        df = self.data
        tickers = df.columns.get_level_values(0).to_list()
        tickers = [el for el in tickers if el not in self.to_blacklist]
        ohlc_ml = list(set(df.columns.get_level_values(1)))

        # get features
        self.dt_bl = df.loc[:,df.columns.get_level_values(0).isin(tickers)]
        df = self.dt_bl
        df = df.stack(level=0)
        df['Pct'] = (df['High']-df['Low'])/df['Close'] * 100.0
        df['PctC'] = (df['Close']-df['Open'])/df['Open'] * 100.0
        df = df.unstack().swaplevel(0,1,axis=1).sort_index(axis=1)

        forecast_out = int(math.ceil(self.forecast_out * len(df)))
        forecast_col = 'Adj Close'

        # df = df.stack(level=0)
        # print(df.groupby(level=1))
        # df['label'] = df.groupby(level=0)
        # print(df)
        # df = df.unstack().swaplevel(0,1,axis=1).sort_index(axis=1)
        # print(df.iloc[:,df.columns.get_level_values(0)=='AAPL'])
        # print(df.iloc[:,df.columns.get_level_values(1)=='Adj Close'].shift(-10).stack(level=1))
        # df = df.stack(level=1)
        # print(df[df.index.get_level_values(1)=='Adj Close'].shift(-forecast_out))
        # df = df.unstack().swaplevel(0,1,axis=1).sort_index(axis=1)
        # print(df['AAPL'])

        
