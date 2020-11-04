#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd 
import warnings

from dt_clustering import Clustering
from dt_help import Helper
from dt_ml import MLPred
from dt_pdr import HistData
from dt_portfolio import Portfolio
from dt_strat import Strat
from pandas.plotting import register_matplotlib_converters

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()

if __name__ == '__main__':
    obj_helper = Helper('data_in','conf_help.yml')
    obj_helper.read_prm()
    
    fontsize = obj_helper.conf['font_size']
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['text.color'] = 'k'

    data_obj = HistData('data_in','data_out','conf_pdr.yml')
    data_obj.read_prm()
    data_obj.process()
    
    # clutering_obj = Clustering('data_in','data_out','conf_clustering.yml',
    #                            data_obj.dt_clustering_ret,
    #                            data_obj.dt_clustering_raw,
    #                            data_obj.dt_raw_cap)

    # clutering_obj.read_prm()
    # clutering_obj.get_clusters_hac()
    # clutering_obj.get_clusters_dbscan_search()
    # clutering_obj.get_clusters_dbscan()

    # strat_obj = Strat('data_in','data_out','conf_strat.yml',
    #                   data_obj.dt_clustering_raw,
    #                   clutering_obj.final_pairs)
    # strat_obj.read_prm()
    # strat_obj.strat_ma()

    # Helper.risk_return(data_obj.dt_clustering_ret)
    # ml_pred = MLPred('data_in','data_out','conf_ml.yml',data_obj.dt_ml)
    # ml_pred.read_prm()
    # ml_pred.ml_model()

    # portfolio_obj = Portfolio('data_in','data_out','conf_portfolio.yml',ml_pred.dt_bl)
    # portfolio_obj.read_prm()
    # portfolio_obj.show_frontier_simple()
