#!/usr/bin/python3

import alphavantage
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import requests_cache
import yaml

from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from dt_help import Helper
from scipy.stats import norm,shapiro,skew,kurtosis
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
        self.urls = self.conf.get('urls')
        self.index = self.conf.get('index')
        self.fetch_data = self.conf.get('fetch_data')
        self.freq = self.conf.get('freq')
        self.market_cap = self.conf.get('market_cap')
        self.features = self.conf.get('features')
        self.new_db_tickers = self.conf.get('new_db_tickers')
        self.new_db_raw_cap = self.conf.get('new_db_raw_cap')
        self.new_db_raw_prices = self.conf.get('new_db_raw_prices')
        self.new_db_alphav_prices = self.conf.get('new_db_alphav_prices')

    @Helper.timing
    def process(self):
        # get the tickers for the selected index
        HistData.get_tickers(self)
        
        # get the tickers based on selected market cap threshold value
        HistData.get_mrkt_cap(self)

        # get raw prices
        HistData.get_raw_prices(self)
        
        # removing tickers that could not be downloaded from pandas datareader
        self.dt_raw_prices = self.dt_raw_prices.dropna(axis=1)
        
        # get the relevant features for the ml analysis
        if(self.fetch_data=='pdr'):
            self.dt_ml = self.dt_raw_prices[self.ohlc_ml]
        elif(self.fetch_data=='alphav'):
            self.dt_ml = self.dt_raw_prices.loc[:,self.dt_raw_prices.columns.get_level_values(1).isin(self.ohlc_ml)]

        # retrieve tickers with missing values/NaN (beyond those that showed no data from the beginning)
        # check performed on prices (if no prices, it follows that Volume will also have issue)
        if(self.fetch_data=='pdr'):
            self.missing_value_tickers = Helper.missing_values_table(self.dt_select[self.ohlc])['Missing Value Tickers']
        elif(self.fetch_data=='alphav'):
            df = self.dt_select.loc[:,self.dt_select.columns.get_level_values(1).isin([self.ohlc])]
            df.columns = df.columns.droplevel(1)
            self.missing_value_tickers = Helper.missing_values_table(df)['Missing Value Tickers']

        # here we decide to drop those tickers that have missing values as it will most likely impact the clustering/DBSCAN
        # the idea is that we want to have consistency in the form of the time series (no flat zeros and then data or drop dates which
        # will reduce the sample necessary for the analysis) 
        if(self.fetch_data=='pdr'):
            self.dt_select[self.ohlc].drop(columns=self.missing_value_tickers,inplace=True)
            self.dt_select['Volume'].drop(columns=self.missing_value_tickers,inplace=True)
        elif(self.fetch_data=='alphav'):
            self.dt_select[self.ohlc].drop(columns=self.missing_value_tickers,inplace=True)
            self.dt_select['Volume'].drop(columns=self.missing_value_tickers,inplace=True)
            
        # final tickers given the dropped missing value tickers
        final_tickers = list(set(self.dt_select[self.ohlc].columns) - set(self.missing_value_tickers))

        # sanity check
        assert(set(final_tickers) - set(list(self.dt_select[self.ohlc].columns))==set())

        # compute returns as (P_t-P_{t-1})/P_{t-1} along with log rturns too
        self.dt_select = self.dt_select.join(pd.DataFrame(self.dt_select[self.ohlc].apply(lambda x: x.pct_change().fillna(0)).values,
                                  columns=pd.MultiIndex.from_product([['rel_ret'], self.dt_select[self.ohlc].columns]),
                                  index=self.dt_select.index))
        self.dt_select = self.dt_select.join(pd.DataFrame(self.dt_select[self.ohlc].apply(lambda x: np.log(x).diff().fillna(0)).values,
                                  columns=pd.MultiIndex.from_product([['log_ret'], self.dt_select[self.ohlc].columns]),
                                  index=self.dt_select.index))

        self.denoise_detrend = Helper.get_denoise_detrend(self.dt_select,'rel_ret')
        
        self.dt_skew = [ Helper.skew_group(self.denoise_detrend,'weekly','denoise_detrend'), Helper.skew_group(self.denoise_detrend,'monthly','denoise_detrend') ]
        Helper.skew_plt(self.dt_skew[0],self.dt_skew[1])
        
        # isolate dataframes used for the clustering step
        self.dt_clustering_ret = self.dt_select['rel_ret']
        self.dt_clustering_raw = self.dt_select

        # Helper.plot_cumulative_ret(self.dt_select)
        # Helper.plot_cumulative_log_ret(self.dt_select)

        # compute the Intrinsic Mode Functions of the price for
        # each ticker these will be used in the second strategy
        # res = []
        # for el in final_tickers:
        #     df = self.dt_select[[el]]
        #     res.append(Helper.get_imfs_hilbert_ts(df,el))
        # self.df_all_ticks_imfs = pd.concat(res,axis=1)

    @Helper.timing
    def get_tickers(self):
        if(self.index == 0):
            tickers_names = Helper.nasdaq100_tickers(self.urls[self.index])
            tickers_names = tickers_names
        elif(self.index == 1):
            tickers_names = Helper.sp500_tickers(self.urls[self.index])

        if(self.new_db_tickers):
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

    @Helper.timing
    def get_mrkt_cap(self):
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
        self.selected_tickers = [el.replace('\n','') for el in selected_tickers]
        
    @Helper.timing
    def get_raw_prices(self):
        if(self.fetch_data == 'pdr'):
            self.attrs = ['Adj Close','Close','High','Open','Low','Volume']
            self.symbs = self.tickers_names.index
            self.midx = pd.MultiIndex.from_product([self.attrs,self.symbs],names=('Attributes', 'Symbols'))

            if(self.new_db_raw_prices):
                dt_raw_prices = pdr.DataReader(self.tickers_names,'yahoo',self.start_date,self.end_date)
                dt_raw_prices['Dates'] = pd.to_datetime(dt_raw_prices.index,format='%Y-%m-%d')

                engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_prices_pdr.db", echo=False)
                dt_raw_prices.to_sql(
                    'dt_raw_prices',
                    engine,
                    if_exists='replace',
                    index=False
                    )
            else:
                engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_prices_pdr.db", echo=False)

            self.dt_raw_prices = pd.read_sql_table(
                'dt_raw_prices',
                con=engine,
                parse_dates={'Dates': {'format': '%Y-%m-%d'}}
                )

            self.dt_raw_prices.rename(columns={"('Dates', '')":'Dates'},inplace=True)
            self.dt_raw_prices.set_index('Dates',inplace=True)
            self.dt_raw_prices.columns = self.midx           
            self.dt_select = self.dt_raw_prices[[self.ohlc,'Volume']]

            # print(self.dt_raw_prices)
            # print(self.dt_select)
            
        elif(self.fetch_data == 'alphav'):
            self.symbs = ['NVDA','AMZN','TSLA','MRNA','AAPL'] # Helper.nasdaq100_tickers(self.url_nasdaq).index.to_list()

            if(self.freq == 'monthly'):
                self.attrs = ['Open','High','Low','Close','Adj Close','Volume','Dividend']
            elif(self.freq == 'daily'):
                self.attrs = ['Open','High','Low','Close','Adj Close','Volume','Dividend','Split coeff']
            elif(self.freq == 'intraday'):
                self.attrs = ['Open','High','Low','Close','Volume']

            self.midx = pd.MultiIndex.from_product([self.symbs,self.attrs],names=('Symbols','Attributes'))

            if(self.new_db_alphav_prices):
                api_key = os.getenv('ALPHAVANTAGE_API_KEY')
                ts = TimeSeries(key=api_key, output_format='pandas')
                
                data_all = []; meta_data_all = []

                if(self.freq == 'monthly'):
                    for i, el in enumerate(self.symbs):
                        data, meta_data = ts.get_monthly_adjusted(symbol=el)
                        data.columns = self.attrs
                        data_all.append(data)
                        meta_data_all.append(meta_data)
                elif(self.freq == 'daily'):
                    for i, el in enumerate(self.symbs):
                        data, meta_data = ts.get_daily_adjusted(symbol=el)
                        data.columns = self.attrs
                        data_all.append(data)
                        meta_data_all.append(meta_data)
                elif(self.freq == 'intraday'):
                    for i, el in enumerate(self.symbs):
                        data, meta_data = ts.get_intraday(symbol=el,interval='60min',outputsize='compact')
                        data.columns = self.attrs
                        data_all.append(data)
                        meta_data_all.append(meta_data)

                self.dt_raw_prices = pd.concat(data_all,axis=1)
                self.dt_raw_prices.columns = self.midx
                self.dt_raw_prices = self.dt_raw_prices[::-1]
                self.dt_raw_prices['Dates'] = pd.to_datetime(self.dt_raw_prices.index,format='%Y-%m-%d')
                
                engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_prices_alv.db", echo=False)
                self.dt_raw_prices.to_sql(
                    'dt_raw_prices',
                    engine,
                    if_exists='replace',
                    index=False
                    )
            else:
                engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_prices_alv.db", echo=False)

            self.dt_raw_prices = pd.read_sql_table(
                'dt_raw_prices',
                con=engine,
                parse_dates={'Dates': {'format': '%Y-%m-%d'}}
                )

            self.dt_raw_prices.rename(columns={"('Dates', '')":'Dates'},inplace=True)
            self.dt_raw_prices.set_index('Dates',inplace=True)
            self.dt_raw_prices.columns = self.midx
            self.dt_raw_prices.index = pd.to_datetime(self.dt_raw_prices.index)
            self.dt_raw_prices = self.dt_raw_prices.swaplevel(0, 1, 1).sort_index(1)
            
            # select the tickers based on ohlc parameters
            if(self.freq == 'monthly' or self.freq == 'daily'):
                self.dt_select = self.dt_raw_prices.loc[:,self.dt_raw_prices.columns.get_level_values(0).isin([self.ohlc]+['Volume'])]
            elif(self.freq == 'intraday'):
                self.dt_select = self.dt_raw_prices.loc[:,self.dt_raw_prices.columns.get_level_values(0).isin(['Close','Volume'])]

            # print(self.dt_raw_prices)
            # print(self.dt_select)
                
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


        

        

