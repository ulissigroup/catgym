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

class Callback():
    def __init__(self, log_dir=None, plot_frequency=50):
        self.log_dir = log_dir
        self.plot_frequency = plot_frequency
    
#     def plot_energy(self, results, xlabel, ylabel, save_path):
#         energies = np.array(results['energies'])
#         actions = np.array(results['actions'])
#         minima_energies = results['minima_energies']
#         minima_steps = results['minima_steps']
#         TS_energies = results['TS_energies']
#         TS_steps = results['TS_steps']
        
#         timesteps = np.arange(len(energies))
#         transition_state_search = np.where(actions==2)[0]
        
#         plt.figure(figsize=(9, 7.5))
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.title(xlabel+ ' vs. ' + ylabel)
        
#         plt.plot(energies, color='black')

#         for action_index in range(len(ACTION_LOOKUP)):
#             action_time = np.where(actions==action_index)[0]
#             plt.plot(action_time, energies[action_time], 'o', 
#                     label=ACTION_LOOKUP[action_index])
        
#         plt.scatter(minima_steps, minima_energies, label='minima', marker='x', color='black', s=150)
#         plt.scatter(TS_steps, TS_energies, label='TS', marker='x', color='r', s=150)
        
#         plt.legend(loc='upper left')
#         plt.savefig(save_path, bbox_inches = 'tight')
#         return plt.close('all')
    
    def plot_summary(self, plotting_values, xlabel, ylabel, save_path):
        plt.figure(figsize=(9, 7.5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(ylabel+ ' vs. ' + xlabel)
        plt.plot(plotting_values)
        plt.savefig(save_path, bbox_inches = 'tight')
        return plt.close('all')       
    
    def episode_finish(self, runner, parallel):  
        log_dir = os.path.join(self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        results = {}
        results['episode'] = runner.episodes
        results['reward'] = runner.episode_reward
        results['updates'] = runner.updates
        results['actions'] = runner.agent.actions_buffers['action_type'].tolist()
        for key in runner.agent.states_buffers:
            if key != 'action_type_mask' and key != 'atom_selection_mask':
                results[key] = runner.agent.states_buffers[key].tolist()
       
        rewards = runner.episode_rewards
        running_times = runner.episode_agent_seconds
        
        with open(os.path.join(log_dir, 'rewards.txt'), 'w') as outfile:
            json.dump(rewards, outfile)
        reward_path = os.path.join(log_dir, 'rewards.png')
        time_path = os.path.join(log_dir, 'running_times.png')

        self.plot_summary(rewards, 'episodes', 'reward', reward_path)
        self.plot_summary(running_times, 'episodes', 'seconds', time_path)
                        
        results_dir = os.path.join(log_dir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with open(os.path.join(results_dir, 'results_%d.txt' %results['episode']), 'w') as outfile:
            json.dump(results, outfile)                       
        
        if results['episode'] % self.plot_frequency == 0: 
            episode_dir = os.path.join(log_dir, 'episode_%d' %results['episode'])
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir)
    
##### TODO: Find a way to save energy and trajectories (use gym_recorder or modifiy runner.py)

#             energy_path = os.path.join(episode_dir, 'energy_%d.png' %results['episode'])
#             self.plot_energy(results, 'steps', 'energy', energy_path)
        
#             trajectories = []
#             for atoms in env.trajectories:
#                 atoms.set_calculator(EMT())
#                 trajectories.append(atoms)
#             write(os.path.join(episode_dir, 'episode_%d.traj' %results['episode']), trajectories)

        return True