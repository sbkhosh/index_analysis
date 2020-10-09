#!/usr/bin/python3

import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import requests_cache
import yaml

from datetime import datetime, timedelta
from dt_help import Helper
from sqlalchemy import create_engine

class HistData():
    def __init__(self,input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

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
        self.start_date = self.conf.get('start_date')
        self.end_date = self.conf.get('end_date')
        self.ohlc = self.conf.get('ohlc')
        self.ohlc_ml = self.conf.get('ohlc_ml')
        self.url_nasdaq = self.conf.get('url_nasdaq')
        self.url_spx = self.conf.get('url_spx')
        self.market_cap = self.conf.get('market_cap')
        self.features = self.conf.get('features')
        self.new_db_tickers = self.conf.get('new_db_tickers')
        self.new_db_raw_cap = self.conf.get('new_db_raw_cap')
        self.new_db_raw_prices = self.conf.get('new_db_raw_prices')
        
    @Helper.timing
    def process(self):
        start_date = self.start_date
        end_date = self.end_date
       
        # get the tickers for Nasdaq100
        if(self.new_db_tickers):
            tickers_names = Helper.nasdaq100_tickers(self.url_nasdaq)
            engine = create_engine("sqlite:///" + self.output_directory + "/tickers_names.db", echo=False)
            tickers_names.to_sql(
                'tickers_names',
                engine,
                if_exists='replace',
                index=True,
                )
        else:
            engine = create_engine("sqlite:///" + self.output_directory + "/tickers_names.db", echo=False)

        self.tickers_names = pd.read_sql_table(
            'tickers_names',
            con=engine
            ).set_index('tickers')

        # get the features for tickers
        if(self.new_db_raw_cap):
            dt_raw_cap = pdr.data.get_quote_yahoo(list(self.tickers_names.index))
            engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_cap.db", echo=False)
            dt_raw_cap.to_sql(
                'dt_raw_cap',
                engine,
                if_exists='replace',
                index=False,
                )
        else:
            engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_cap.db", echo=False)

        self.dt_raw_cap = pd.read_sql_table(
            'dt_raw_cap',
            con=engine
            )
        self.dt_raw_cap.index = self.tickers_names.index

        # get only a limited number of features for each ticker
        self.dt_raw_cap = self.dt_raw_cap[self.features]
        
        # transform market cap to billions        
        self.dt_raw_cap['marketCap'] = self.dt_raw_cap['marketCap'] / 1e9
        
        # select on market cap (threshold set in yaml parameter file)
        selected_tickers = list(self.dt_raw_cap[self.dt_raw_cap['marketCap'] > self.market_cap].index)
        selected_tickers = [el.replace('\n','') for el in selected_tickers]

        attrs = ['Adj Close','Close','High','Open','Low','Volume']
        symbs = selected_tickers
        midx = pd.MultiIndex.from_product([attrs,symbs],names=('Attributes', 'Symbols'))
        
        if(self.new_db_raw_prices):
            dt_raw_prices = pdr.DataReader(selected_tickers,'yahoo',start_date,end_date)
            dt_raw_prices['Dates'] = pd.to_datetime(dt_raw_prices.index,format='%Y-%m-%d')
            
            engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_prices.db", echo=False)
            dt_raw_prices.to_sql(
                'dt_raw_prices',
                engine,
                if_exists='replace',
                index=False
                )
        else:
            engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_prices.db", echo=False)

        self.dt_raw_prices = pd.read_sql_table(
            'dt_raw_prices',
            con=engine,
            parse_dates={'Dates': {'format': '%Y-%m-%d'}}
            )

        self.dt_raw_prices.rename(columns={"('Dates', '')":'Dates'},inplace=True)
        self.dt_raw_prices.set_index('Dates',inplace=True)
        self.dt_raw_prices.columns = midx

        # this removes that tickers that could not be downloaded from pandas datareader
        self.dt_raw_prices = self.dt_raw_prices.dropna(axis=1)
        
        # select the tickers based on ohlc parameters
        res = self.dt_raw_prices.loc[:,self.dt_raw_prices.columns.get_level_values(0).isin(self.ohlc)]
        self.dt_select = res
        self.dt_ml = self.dt_raw_prices.loc[:,self.dt_raw_prices.columns.get_level_values(0).isin(self.ohlc_ml)]

        print(isinstance(self.dt_select.columns,pd.MultiIndex))
        # print(self.dt_select.isnull().sum().groupby('Symbols'))
        # retrieve tickers with missing values/NaN (beyond those that showed no data from the beginning)
        # self.missing_value_tickers = Helper.missing_values_table(self.dt_select)['Missing Value Tickers']
        
        # # # here we decide to drop those tickers that have missing values as it will most likely impact the clustering/DBSCAN
        # # # the idea is that we want to have consistency in the form of the time series (no flat zeros and then data or drop dates which
        # # # will reduce the sample necessary for the analysis) 
        # self.dt_select.drop(columns=self.missing_value_tickers,inplace=True)

        # # final tickers given the dropped missing value tickers
        # final_tickers = list(set(self.dt_select.columns) - set(self.missing_value_tickers))
        
        # # sanity check
        # assert(set(final_tickers) - set(list(self.dt_select.columns))==set())

        # # compute returns as (P_t-P_{t-1})/P_{t-1} along with log rturns too
        # self.dt_select[[el+'_rel_ret' for el in final_tickers]] = self.dt_select[final_tickers].apply(lambda x: x.pct_change().fillna(0))
        # self.dt_select[[el+'_log_ret' for el in final_tickers]] = self.dt_select[final_tickers].apply(lambda x: np.log(x).diff().fillna(0))

        # # isolate dataframes used for the clustering step
        # self.dt_clustering_ret = self.dt_select[[el+'_rel_ret' for el in final_tickers]]
        # self.dt_clustering_ret.columns = final_tickers
        # self.dt_clustering_raw = self.dt_select[[el for el in final_tickers]]

        # # sanity check => no missing values in the returns series
        # assert(len(Helper.missing_values_table(self.dt_select[[el+'_rel_ret' for el in final_tickers]]).values) == 0)

        # Helper.plot_cumulative_ret(self.dt_select)
        # Helper.plot_cumulative_log_ret(self.dt_select)

        # compute the Intrinsic Mode Functions of the price for
        # each ticker these will be used in the second strategy
        # res = []
        # for el in final_tickers:
        #     df = self.dt_select[[el]]
        #     res.append(Helper.get_imfs_hilbert_ts(df,el))
        # self.df_all_ticks_imfs = pd.concat(res,axis=1)
               
    def get_info_1d(cum_rets):
        end = np.argmax((np.maximum.accumulate(cum_rets)-cum_rets)/np.maximum.accumulate(cum_rets))
        start = np.argmax(cum_rets[:end])
        mdd = np.round(cum_rets[end]-cum_rets[start],2) * 100
        mdd_duration = (cum_rets.index[end]-cum_rets.index[start]).days
        start_date = cum_rets.index[start]
        end_date = cum_rets.index[end]
        cagr = (cum_rets[-1]/cum_rets[1]) ** (252.0 / len(cum_rets)) - 1

        return({'mdd': mdd,'mdd_duration': mdd_duration,
                'start': int(start), 'end': int(end),
                'start_date': start_date , 'end_date': end_date,
                'cum_ret_start': cum_rets[start_date], 'cum_ret_end': cum_rets[end_date],
                'cagr': cagr}) 


        

        

