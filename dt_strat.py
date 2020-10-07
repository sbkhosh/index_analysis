#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import yaml

from datetime import datetime, timedelta
from dt_help import Helper
from PyEMD import EMD

class Strat():
    def __init__(self,
                 input_directory, output_directory, input_prm_file,
                 data: pd.DataFrame,
                 tickers_pair_strat):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file
        self.data = data
        self.tickers_pair_strat = tickers_pair_strat
        
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
        self.short_ma = self.conf.get('short_ma')
        self.middle_ma = self.conf.get('middle_ma')
        self.long_ma = self.conf.get('long_ma')
        self.initial_capital = self.conf.get('initial_capital')
        self.sample_ratio = self.conf.get('sample_ratio')
        self.chosen_ticker = self.conf.get('chosen_ticker')
        
    @Helper.timing
    def strat_ewma(self):
        df = self.data
        df_in_sample = Helper.get_in_out_sample(df,'in',self.sample_ratio)
        df_out_sample = Helper.get_in_out_sample(df,'out',self.sample_ratio)
        tickers = df.columns.values
        
        signals = pd.DataFrame()
        signals[[el+'_short' for el in tickers]] = np.round(df_in_sample[[el for el in tickers]].ewm(span=self.short_ma).mean(),2)
        signals[[el+'_middle' for el in tickers]] = np.round(df_in_sample[[el for el in tickers]].ewm(span=self.middle_ma).mean(),2)
        signals[[el+'_long' for el in tickers]] = np.round(df_in_sample[[el for el in tickers]].ewm(span=self.long_ma).mean(),2)
        df_all = pd.concat([df_in_sample,data_ewma],axis=1)
        df = df_all[[el for el in df_all.columns if self.chosen_ticker in el]]

        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0
        signals = pd.concat([df_in_sample,signals],axis=1)

    @Helper.timing
    def strat_ma(self):
        df = Helper.get_in_out_sample(self.data[[self.chosen_ticker]],'in',self.sample_ratio)

        signals = pd.DataFrame()
        signals['signal'] = 0.0
        
        signals[self.chosen_ticker+'_short_ma'] = np.round(df[self.chosen_ticker].rolling(window=self.short_ma).mean(),2)
        signals[self.chosen_ticker+'_long_ma'] = np.round(df[self.chosen_ticker].rolling(window=self.long_ma).mean(),2)
        
        signals['signal'][self.short_ma:] = np.where(signals[self.chosen_ticker+'_short_ma'][self.short_ma:] > signals[self.chosen_ticker+'_long_ma'][self.short_ma:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()

        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions[self.chosen_ticker] = 100*signals['signal']   # This strategy buys 100 shares

        portfolio = positions*df[[self.chosen_ticker]]
        pos_diff = positions.diff()

        portfolio['holdings'] = (positions[[self.chosen_ticker]]*df[[self.chosen_ticker]]).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff[[self.chosen_ticker]]*df[[self.chosen_ticker]]).sum(axis=1).cumsum()
        
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['rel_ret'] = portfolio['total'].pct_change()
        portfolio['log_ret'] = np.log(portfolio['total']).diff().fillna(0)

        self.signals = signals
        self.portfolio = portfolio
        self.in_sample_data = df
        # Strat.plot_eq_curve(self)
        # Strat.plot_summary_curves(self)
        
    @Helper.timing
    def plot_eq_curve(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(32,20))

        # cumulative return from relative returns definition
        self.portfolio[[el for el in self.portfolio if 'rel_ret' in el]].apply(lambda x: (1.0 + x).cumprod()-1.0).plot(ax=ax1)
        ax1.set_ylabel('Cumulative returns (1st def)')
        ax1.legend(bbox_to_anchor=(2, 1), loc='upper right', ncol=1)

        # second way of computing from log returns (as a check)
        self.portfolio[[el for el in self.portfolio.columns if 'log_ret' in el]].apply(lambda x: (np.exp(x.cumsum()) - 1)).plot(ax=ax2)
        ax2.set_ylabel('Cumulative returns (2nd def)')
        ax2.legend().set_visible(False)

        self.portfolio[['total']].plot(ax=ax3)
        ax3.set_ylabel('Cumulative sum of portfolio')
        ax3.legend().set_visible(False)
        plt.show()

