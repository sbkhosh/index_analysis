#!/usr/bin/python3

import alpha_vantage
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader as pdr
import requests_cache
import yaml

from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from dt_help import Helper
from sqlalchemy import create_engine

class Portfolio():
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
        self.rr_symb1 = self.conf.get('rr_symb1')
        self.rr_symb2 = self.conf.get('rr_symb2')
        self.new_db_alphav_prices = self.conf.get('new_db_alphav_prices')
        self.freq = self.conf.get('freq')
            
    @Helper.timing
    def show_frontier_simple(self):
        symbs = [self.rr_symb1,self.rr_symb2]

        if(self.freq == 'monthly'):
            attrs = ['Open','High','Low','Close','Adj Close','Volume','Dividend']
        elif(self.freq == 'daily'):
            attrs = ['Open','High','Low','Close','Adj Close','Volume','Dividend','Split coeff']
            
        midx = pd.MultiIndex.from_product([symbs,attrs],names=('Symbols','Attributes'))

        if(self.new_db_alphav_prices):
            api_key = os.getenv('ALPHAVANTAGE_API_KEY')
            ts = TimeSeries(key=api_key, output_format='pandas')

            if(self.freq == 'monthly'):
                data1, meta_data1 = ts.get_monthly_adjusted(symbol=self.rr_symb1)
                data2, meta_data2 = ts.get_monthly_adjusted(symbol=self.rr_symb2)
            elif(self.freq == 'daily'):
                data1, meta_data1 = ts.get_daily_adjusted(symbol=self.rr_symb1)
                data2, meta_data2 = ts.get_daily_adjusted(symbol=self.rr_symb2)
                
            data1.columns = attrs
            data2.columns = attrs

            dt_alphav_prices = pd.concat([data1,data2],axis=1)
            dt_alphav_prices.columns = midx
            dt_alphav_prices = dt_alphav_prices[::-1]
            dt_alphav_prices['Dates'] = pd.to_datetime(dt_alphav_prices.index,format='%Y-%m-%d')
            
            engine = create_engine("sqlite:///" + self.output_directory + "/dt_alphav_prices.db", echo=False)
            dt_alphav_prices.to_sql(
                'dt_alphav_prices',
                engine,
                if_exists='replace',
                index=False
                )
        else:
            engine = create_engine("sqlite:///" + self.output_directory + "/dt_alphav_prices.db", echo=False)

        self.dt_alphav_prices = pd.read_sql_table(
            'dt_alphav_prices',
            con=engine,
            parse_dates={'Dates': {'format': '%Y-%m-%d'}}
            )

        self.dt_alphav_prices.rename(columns={"('Dates', '')":'Dates'},inplace=True)
        self.dt_alphav_prices.set_index('Dates',inplace=True)
        self.dt_alphav_prices.columns = midx

        returns1 = self.dt_alphav_prices[(self.rr_symb1,'Adj Close')].pct_change().fillna(0)
        returns2 = self.dt_alphav_prices[(self.rr_symb2,'Adj Close')].pct_change().fillna(0)

        if(len(returns1) > len(returns2)):
            returns1 = returns1[-len(returns2):]

        if(len(returns2) > len(returns1)):
            returns2 = returns2[-len(returns1):]

        mean_returns1 = np.mean(returns1)
        variance1 = np.var(returns1)
        standard_deviation1 = np.sqrt(variance1)

        mean_returns2 = np.mean(returns2)
        variance2 = np.var(returns2)
        standard_deviation2 = np.sqrt(variance2)

        correlation = np.corrcoef(returns1, returns2)[0][1]
        print(f'Correlation = {correlation}')

        weights = []
        for n in range(0, 101):
            weights.append((1 - 0.01 * n, 0 + 0.01 * n))
            
        returns = []
        standard_deviations = []

        portfolio_50_50_standard_deviation = None
        portfolio_50_50_returns = None

        plt.figure(figsize=(32,20))
        for w1, w2 in weights:
            returns.append(w1 * mean_returns1 + w2 * mean_returns2)

            variance = w1**2 * standard_deviation1**2 + w2**2 * standard_deviation2**2 + \
                2 * w1 * w2 * standard_deviation1 * standard_deviation2 * correlation

            standard_deviation = np.sqrt(variance)
            standard_deviations.append(standard_deviation)

            plt.scatter(standard_deviations[-1], returns[-1], color='#007bff')

            if w1 == 0.5 and w2 == 0.5:
                portfolio_50_50_standard_deviation = standard_deviations[-1]
                portfolio_50_50_returns = returns[-1]

        plt.scatter(portfolio_50_50_standard_deviation,
                    portfolio_50_50_returns, marker='x', color='red', alpha=1, s=320)

        x_padding = np.average(standard_deviations) / 25
        plt.xlim(min(standard_deviations) - x_padding,
                 max(standard_deviations) + x_padding)

        y_padding = np.average(returns) / 25
        plt.ylim(min(returns) - y_padding, max(returns) + y_padding)

        plt.gca().set_xticks(plt.gca().get_xticks().tolist()) # remove in the future - placed to avoid warning - it is a bug from matplotlib 3.3.1
        plt.gca().set_xticklabels(['{:.2f}%'.format(x*100) for x in plt.gca().get_xticks()])
        plt.gca().set_yticks(plt.gca().get_yticks().tolist()) # remove in the future - placed to avoid warning - it is a bug from matplotlib 3.3.1
        plt.gca().set_yticklabels(['{:.2f}%'.format(y*100) for y in plt.gca().get_yticks()])

        plt.title(f'Efficient Frontier ({self.rr_symb1.upper()} and {self.rr_symb2.upper()})')

        plt.xlabel('Risk (' + self.freq.capitalize() + ')')
        plt.ylabel('Return (' + self.freq.capitalize() + ')')
        plt.show()
