import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from surface_seg.envs.mcs_env import ACTION_LOOKUP
from ase.io import write
from asap3 import EMT

class Callback():
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
    
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

    def plot_rewards(self, rewards, xlabel, ylabel, save_path):
        plt.figure(figsize=(9, 7.5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(xlabel+ ' vs. ' + ylabel)
        plt.plot(rewards)
        plt.savefig(save_path, bbox_inches = 'tight')
        return plt.close('all')

    def episode_finish(self, runner, parallel):  
        results_dir = os.path.join(self.log_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        env = runner.environments[0].environment.environment.env
        
        results = {}
        results['episode'] = runner.episodes
        results['reward'] = runner.episode_reward[0]
        results['updates'] = runner.updates
        results['initial_energy'] = env.initial_energy
        results['energies'] = env.energies
        results['actions'] = env.actions
        results['minima_energies'] = env.minima['energies']
        results['minima_steps'] = env.minima['timesteps']
        results['TS_energies'] = env.TS['energies']
        results['TS_steps'] = env.TS['timesteps']
        
        
        rewards = runner.episode_rewards
        with open(os.path.join(results_dir, 'rewards.txt'), 'w') as outfile:
            json.dump(rewards, outfile)
        reward_path = os.path.join(results_dir, 'rewards.png')

        self.plot_rewards(rewards, 'episodes', 'reward', reward_path)

        episode_dir = os.path.join(results_dir, 'episode_'+str(runner.episodes))
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        energy_path = os.path.join(episode_dir, 'energies.png')
        


        self.plot_energy(results, 'steps', 'energy', energy_path)

        with open(os.path.join(episode_dir, 'results.txt'), 'w') as outfile:
            json.dump(results, outfile)    
            
        trajectories = []
        for atoms in env.trajectories:
            atoms.set_calculator(EMT())
            trajectories.append(atoms)
        
        write(os.path.join(episode_dir, 'trajectories.traj'), trajectories)

        return True