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
            
    def show_frontier(self):
        df = self.data
        df.index = pd.to_datetime(df.index)
        returns1 = df[self.rr_symb1]['Adj Close'].pct_change().fillna(0)
        returns2 = df[self.rr_symb2]['Adj Close'].pct_change().fillna(0)

        # returns1 = returns1.groupby((returns1.index.month)).apply(lambda x: (1+x).prod()-1)
        # returns2 = returns2.groupby((returns2.index.month)).apply(lambda x: (1+x).prod()-1)
        
        if(len(returns1) > len(returns2)):
            returns1 = returns1[-len(returns2):]

        if(len(returns2) > len(returns1)):
            returns2 = returns2[-len(returns1):]

        mean_returns1 = np.mean(returns1)
        variance1 = np.var(returns1)
        standard_deviation1 = np.sqrt(variance1)

        #print(f'Mean returns ({symbol1}) = {mean_returns1}')
        #print(f'Variance ({symbol1}) = {variance1}')
        #print(f'Standard Deviation ({symbol1}) = {standard_deviation1}')

        mean_returns2 = np.mean(returns2)
        variance2 = np.var(returns2)
        standard_deviation2 = np.sqrt(variance2)

        #print(f'Mean returns ({symbol2}) = {mean_returns2}')
        #print(f'Variance ({symbol2}) = {variance2}')
        #print(f'Standard Deviation ({symbol2}) = {standard_deviation2}')

        correlation = np.corrcoef(returns1, returns2)[0][1]
        print(f'Correlation = {correlation}')

        weights = []

        for n in range(0, 101):
            weights.append((1 - 0.01 * n, 0 + 0.01 * n))
            
        returns = []
        standard_deviations = []

        portfolio_50_50_standard_deviation = None
        portfolio_50_50_returns = None

        # plot_style.scatter()

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

        plt.gca().set_xticklabels(['{:.2f}%'.format(x*100) for x in plt.gca().get_xticks()])
        plt.gca().set_yticklabels(['{:.2f}%'.format(y*100) for y in plt.gca().get_yticks()])

        plt.title(f'Efficient Frontier ({self.rr_symb1.upper()} and {self.rr_symb2.upper()})')

        plt.xlabel(f'Risk (Daily)')
        plt.ylabel(f'Return (Daily)')

        # pathlib.Path('img/frontier2').mkdir(parents=True, exist_ok=True)
        # plt.savefig(f'img/frontier2/{self.rr_symb1.lower()}-{self.rr_symb2.lower()}.png')
        plt.show()
