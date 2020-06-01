import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from surface_seg.envs.mcs_env import ACTION_LOOKUP

class Callback():
    def __init__(self, log_dir):
        self.log_dir = log_dir
    
    def plot_energy(self, runner, energies, actions, xlabel, ylabel, save_path):
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

        results = {}
        results['episode'] = runner.episodes
        results['reward'] = runner.episode_reward[0]
        results['energies'] = runner.agent.states_buffers['energy'].reshape(-1).tolist()
        results['actions'] = runner.agent.actions_buffers['action_type'].reshape(-1).tolist()

        rewards = runner.episode_rewards
        with open(os.path.join(results_dir, 'rewards.txt'), 'w') as outfile:
            json.dump(rewards, outfile)
        reward_path = os.path.join(results_dir, 'rewards.png')

        self.plot_rewards(rewards, 'episodes', 'reward', reward_path)

        episode_dir = os.path.join(results_dir, 'episode_'+str(runner.episodes))
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        energy_path = os.path.join(episode_dir, 'energies.png')
        energies = runner.agent.states_buffers['energy'].reshape(-1)
        actions = runner.agent.actions_buffers['action_type'].reshape(-1)

        self.plot_energy(runner, energies, actions, 'steps', 'energy', energy_path)

        with open(os.path.join(episode_dir, 'results.txt'), 'w') as outfile:
            json.dump(results, outfile)    

        return True