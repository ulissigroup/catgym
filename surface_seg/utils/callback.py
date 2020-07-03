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
    
    def plot_energy(self, results, xlabel, ylabel, save_path):
        energies = np.array(results['energies'])
        actions = np.array(results['actions'])
        minima_energies = results['minima_energies']
        minima_steps = results['minima_steps']
        TS_energies = results['TS_energies']
        TS_steps = results['TS_steps']
        
        timesteps = np.arange(len(energies))
        transition_state_search = np.where(actions==2)[0]
        
        plt.figure(figsize=(9, 7.5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(xlabel+ ' vs. ' + ylabel)
        
        plt.plot(energies, color='black')

        for action_index in range(len(ACTION_LOOKUP)):
            action_time = np.where(actions==action_index)[0]
            plt.plot(action_time, energies[action_time], 'o', 
                    label=ACTION_LOOKUP[action_index])
        
        plt.scatter(minima_steps, minima_energies, label='minima', marker='x', color='black', s=150)
        plt.scatter(TS_steps, TS_energies, label='TS', marker='x', color='r', s=150)
        
        plt.legend(loc='upper left')
        plt.savefig(save_path, bbox_inches = 'tight')
        return plt.close('all')
    
    def plot_summary(self, plotting_values, xlabel, ylabel, save_path):
        plt.figure(figsize=(9, 7.5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(ylabel+ ' vs. ' + xlabel)
        plt.plot(plotting_values)
        plt.savefig(save_path, bbox_inches = 'tight')
        return plt.close('all')
    
    def twinplot_summary(self, first_value, second_value, xlabel, ylabel_1, ylabel_2, save_path):
        fig, ax1 = plt.subplots(figsize=(9, 7.5))
        
        lns1 = ax1.plot(first_value, label = ylabel_1, color = 'royalblue')
        ax1.set_xlabel(xlabel, fontsize=15)
        ax1.set_ylabel(ylabel_1, fontsize=15, color = 'royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)

        ax2 = ax1.twinx()
        lns2 = ax2.plot(second_value, label = ylabel_2, color = 'orange')
        ax2.set_ylabel(ylabel_2, fontsize=15, color = 'orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        plt.yticks(fontsize=12)
        
        # Making two legends
        legend = lns1 + lns2
        labels = [ax.get_label() for ax in legend]
        ax1.legend(legend, labels, loc = 0)
        plt.savefig(save_path, bbox_inches = 'tight')
        return plt.close('all')        
        
    
    def episode_finish(self, runner, parallel):  
        log_dir = os.path.join(self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        env = runner.environments[0].environment.environment.env
        free_atoms = list(set(range(len(env.atoms))) -
                               set(env.atoms.constraints[0].get_indices()))
        
        results = {}
        results['episode'] = runner.episodes
        results['reward'] = runner.episode_reward[0]
        results['force_calls'] = env.force_calls
        results['updates'] = runner.updates
        results['chemical_symbols'] = env.atoms.get_chemical_symbols()
        results['free_atoms'] = free_atoms
        results['initial_energy'] = env.initial_energy
        results['energies'] = env.energies
        results['actions'] = env.actions
        if 'fingerprints' in runner.agent.states_buffers:
            results['fingerprints'] = runner.agent.states_buffers['fingerprints'][0].tolist()
        results['positions'] = env.positions
        results['minima_energies'] = env.minima['energies']
        results['minima_steps'] = env.minima['timesteps']
        results['TS_energies'] = env.TS['energies']
        results['TS_steps'] = env.TS['timesteps']
        
        fps_params = {}
        fps_params['elements'] = env.elements
        fps_params['descriptors'] = env.descriptors
        with open(os.path.join(log_dir, 'fps_params.txt'), 'w') as outfile:
            json.dump(fps_params, outfile)
        
        rewards = runner.episode_rewards
        running_times = runner.episode_agent_seconds
        force_calls = results['force_calls']
        
        with open(os.path.join(log_dir, 'rewards.txt'), 'w') as outfile:
            json.dump(rewards, outfile)
        reward_path = os.path.join(log_dir, 'rewards.png')
        time_path = os.path.join(log_dir, 'running_times.png')
        force_calls_plot_path = os.path.join(log_dir, 'force_calls.png')
        twin_plot_path = os.path.join(log_dir, 'twin.png')
        
        force_calls_path = os.path.join(log_dir, 'force_calls.txt')
        if os.path.exists(force_calls_path):
            f = json.load(open(force_calls_path, 'r'))
            f.append(env.force_calls)
            json.dump(f, open(force_calls_path, 'w'))
            self.plot_summary(f, 'episodes', 'total force calls', force_calls_plot_path)
            
            self.twinplot_summary(rewards, f, 'episodes', 'rewards', 'total force calls', twin_plot_path)
        else:
            json.dump([env.force_calls], open(force_calls_path, 'w'))
            self.plot_summary([env.force_calls], 'episodes', 'total force calls', force_calls_plot_path)
        
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
                
            energy_path = os.path.join(episode_dir, 'energy_%d.png' %results['episode'])
            self.plot_energy(results, 'steps', 'energy', energy_path)
        
            trajectories = []
            for atoms in env.trajectories:
                atoms.set_calculator(EMT())
                trajectories.append(atoms)
            write(os.path.join(episode_dir, 'episode_%d.traj' %results['episode']), trajectories)

        return True