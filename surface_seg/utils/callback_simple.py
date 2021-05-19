import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from surface_seg.envs.mcs_env import ACTION_LOOKUP
from ase.io import write
from asap3 import EMT
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from surface_seg.envs.symmetry_function import make_snn_params


def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

class Callback():
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        
    def plot_summary(self, plotting_values, xlabel, ylabel, save_path): # plotting_values = rewards
        plt.figure(figsize=(9, 7.5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(ylabel+ ' vs. ' + xlabel)
        plt.plot(plotting_values)
        
        window = 25
        if len(plotting_values) > window:
            steps = np.arange(len(plotting_values))
            yMA = movingaverage(plotting_values, window)
            plt.plot(steps[len(steps)-len(yMA):], yMA)
        plt.savefig(save_path, bbox_inches = 'tight')
        return plt.close('all')
    
    def plot_summary_pd(self, plotting_values, xlabel, ylabel, save_path): # plotting_values = rewards
        plt.figure(figsize=(9, 7.5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(ylabel+ ' vs. ' + xlabel)
#         plt.plot(plotting_values)
        
        window = 25
        if len(plotting_values) > window:
            df = pd.DataFrame(plotting_values)
            sma = df.rolling(window, win_type=None).mean().iloc[window:]
            sma_std = df.rolling(window, win_type=None).std().iloc[window:]

            plt.plot(np.arange(len(sma)), sma.to_numpy().reshape(-1), color='r')
            plt.fill_between(np.arange(len(sma)), 
                             (sma-sma_std).to_numpy().reshape(-1), 
                             (sma+sma_std).to_numpy().reshape(-1),
                             alpha=0.5,
                             antialiased=False,
    #                          color='r',
                    )        
        plt.savefig(save_path, bbox_inches = 'tight')
        return plt.close('all')
                 
#     def plot_summary_pd_log(self, plotting_values, xlabel, ylabel, save_path): # plotting_values = rewards
#         plt.figure(figsize=(9, 7.5))
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.title(ylabel+ ' vs. ' + xlabel)
# #         plt.plot(plotting_values)
        
#         window = 25
#         if len(plotting_values) > window:

#             df = pd.DataFrame(plotting_values)
#             sma = df.rolling(window, win_type=None).mean().iloc[window:]
#             sma_std = df.rolling(window, win_type=None).std().iloc[window:]

#             plt.plot(np.arange(len(sma)), sma.to_numpy().reshape(-1), color='r')
#             plt.fill_between(np.arange(len(sma)), 
#                              (sma-sma_std).to_numpy().reshape(-1), 
#                              (sma+sma_std).to_numpy().reshape(-1),
#                              alpha=0.5,
#     #                          color='r',
#                     )        
#             plt.xscale('log')
#         plt.savefig(save_path, bbox_inches = 'tight')
#         return plt.close('all')
    
    def episode_finish(self, runner, parallel):  
        log_dir = os.path.join(self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        rewards = runner.episode_rewards
        
        with open(os.path.join(log_dir, 'rewards.txt'), 'w') as outfile:
            json.dump(rewards, outfile)
            
        reward_path = os.path.join(log_dir, 'rewards.png')
        self.plot_summary(rewards, 'episodes', 'reward', reward_path)
        
        reward_path_pd = os.path.join(log_dir, 'rewards_pd.png')
        self.plot_summary_pd(rewards, 'episodes', 'reward', reward_path_pd)
                 
#         reward_path_pd_log = os.path.join(log_dir, 'rewards_pd_log.png')
#         self.plot_summary_pd(rewards, 'episodes', 'reward', reward_path_pd_log)
        return True
        
    