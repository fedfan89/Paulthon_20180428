import datetime as dt
import pandas as pd
import math
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from collections import namedtuple
from statistics import mean
from paul_resources import InformationTable, tprint, rprint
from decorators import my_time_decorator
from py_vollib.black_scholes.implied_volatility import black_scholes, implied_volatility
from Option_Module import get_time_to_expiry
import logging

# Logging Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(message)s')

file_handler = logging.FileHandler('mc_transform.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

#logging.disable(logging.CRITICAL)

def float_to_distribution(move_input: 'float', csv_file):
    distribution_df = pd.read_csv(csv_file).set_index('State')
    mean_move = Distribution(distribution_df).mean_move

    distribution_df.loc[:, 'Pct_Move'] *= (move_input/mean_move)
    distribution_df.loc[:, 'Relative_Price'] = distribution_df.loc[:, 'Pct_Move'] + 1
        
    return Distribution(distribution_df)

def float_to_event_distribution(move_input: 'float'):
    return float_to_distribution(move_input, 'Event.csv')

def float_to_volbeta_distribution(move_input: 'float'):
    #time_to_expiry = get_time_to_expiry(expiry)
    #return float_to_distribution(move_input*math.sqrt(time_to_expiry), 'VolbetaDistribution.csv')
    return float_to_distribution(move_input, 'VolbetaDistribution.csv')

def float_to_bs_distribution(move_input: 'float'):
    return float_to_distribution(move_input, 'BlackScholes.csv')

def distribution_info_to_distribution(distribution_info):
    distribution_df = pd.DataFrame(distribution_info).set_index('State').loc[:, ['Prob', 'Pct_Move', 'Relative_Price']]
    return Distribution(distribution_df)

def mc_distribution_to_distribution(mc_distribution,
                                    bins = 10**4+1,
                                    to_csv=False,
                                    csv_file_name = None):
    
    mean_mc_price = mean(mc_distribution)
    
    counts, binEdges = np.histogram(mc_distribution, bins)
    binCenters = .5*(binEdges[1:] + binEdges[:-1])
    
    probs = [i / sum(counts) for i in counts]
    relative_moves = [binCenter / mean_mc_price for binCenter in binCenters]
    pct_moves = [relative_move - 1 for relative_move in relative_moves]

    distribution_info = {'State': np.array(range(len(counts))),
                         'Prob': probs,
                         'Pct_Move': pct_moves,
                         'Relative_Price': relative_moves}

    if to_csv is True:
        distribution_df = distribution_info_to_distribution(distribution_info).distribution_df
        distribution_df.to_csv(csv_file_name)
    
    logger.info("Iterations: {:,}".format(sum(counts)))
    logger.info("Total Prob: {:.2f}".format(sum(probs)))
    logger.info("Mean Stock Price: {}".format(mean_mc_price))

    return distribution_info_to_distribution(distribution_info)

if __name__ == '__main__':
    bs_distribution_original = float_to_bs_distribution(.3)
    #print("Original:", bs_distribution_original.mean_move)
    mc_distribution = bs_distribution_original.mc_simulation(10**6)
    bs_distribution_created = mc_distribution_to_distribution(mc_distribution)
    #print(bs_distribution_created.mean_move)

