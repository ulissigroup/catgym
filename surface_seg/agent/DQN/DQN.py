import os
import sys
sys.path.insert(0,'/home/junwoony/Desktop/ARPAE/')

import argparse
import importlib
import json
import logging
import time
import numpy as np

# %env CUDA_DEVICE_ORDER=PCI_BUS_ID
# %env CUDA_VISIBLE_DEVICES=0
# %env CUDA_LAUNCH_BLOCKING=1

import ase.io

from env.surface_env import *
# from utils.terminate import 

from matplotlib import pyplot as plt
from tensorforce import TensorForceError
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
import tensorflow as tf

horizon=200
episodes=10000
env = SurfaceEnv(horizon)
print('Initial energy:', env.get_energy())
seed = 42
np.random.seed(seed)
tf.random.set_random_seed(seed)

network_spec = [
    {
        "type": "dense",
        "size": 64,
        "activation": "relu"
    },
    {
        "type": "dense",
        "size": 32,
        "activation": "relu"
    }
]

agent = DQNAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec,
    batched_observe=True, 
    batching_capacity=8000,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    ), 

    states_preprocessing=None,
    reward_preprocessing=None,

    update_mode=dict(
        unit='timesteps',
        batch_size=10,
        frequency=10
    ),
    memory=dict(
        type='replay',
        include_next_states=True,
        capacity=40000
    ),

    optimizer=dict(
        type='clipped_step',
        clipping_value=0.1,
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        )
    ),
    actions_exploration=dict(
        type='epsilon_anneal',
        initial_epsilon=0.5,
        final_epsilon=0.05,
        timesteps=1000000
    ),
    discount=1,
    distributions=None,
    entropy_regularization=0.01,
    target_sync_frequency=1000,
    target_update_weight=1.0,
    double_q_model=False,
    huber_loss=None,

    summarizer=dict(
        directory=None,
        labels=['graph', 'total-loss']
    ),
)

runner = Runner(
    agent=agent,
    environment=env,
    repeat_actions=1
)


def plot_energy(energy, xlabel, ylabel, save_path):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(xlabel+ ' vs. ' + ylabel)
    plt.plot(energy)
    plt.savefig(save_path)
    print('figure saved as {}'.format(save_path))
    return 

def episode_finished(r):
    # if r.episode % 50 == 0:
    #     positions = env.get_positions()
    #     pos_fn = '_'.join(['pos_3', str(r.episode)])
    #     pos_dir = os.path.join('new_pos', pos_fn)
    #     np.save(pos_dir, positions)
    # if r.episode % 50 == 0:
    #     agent_fn = '_'.join(['agent_3', str(r.episode)])
    #     agent_path = os.path.join('new_agents', agent_fn)
    #     r.agent.save_model(agent_path)
    #     print("Saving agent to {}".format(agent_dir))
    # if r.episode % 50 == 0:
    #     rew_fn = '.'.join(['_'.join(['reward_3', str(r.episode)]), 'png'])
    #     rew_dir = os.path.join('new_plots', rew_fn)
    #     plot_energy(r.episode_rewards, 'accumulated reward', rew_dir)
    #     energy_fn = '.'.join(['_'.join(['final', 'energy_3', str(r.episode)]), 'png'])
    #     energy_dir = os.path.join('new_plots', energy_fn)
    #     plot_energy(env.final_energy, 'final energy', energy_dir)

    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".
        format(ep=r.episode, ts=r.episode_timestep,reward=r.episode_rewards[-1]))
    if r.episode % 1 == 0:
        traj_dir = os.path.join('traj_files', 'seed_'+str(seed), str(r.episode))
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir)
        env.save_traj(traj_dir)

        E_dir = os.path.join('E_figs', 'seed_'+str(seed))
        if not os.path.exists(E_dir):
            os.makedirs(E_dir)
        E_fn = 'E_' + str(r.episode) + '_%f' %env.ts_energy[-1] + '.png'            
        E_fn = os.path.join(E_dir, E_fn)       
        plot_energy(env.energies[:horizon], 'actions', 'energy', E_fn)
#         fig_dir = os.path.join('atom_figs', 'seed_'+str(seed), str(r.episode))
#         if not os.path.exists(fig_dir):
#             os.makedirs(fig_dir)
#         env.save_fig(fig_dir)

    if r.episode % 50 == 0:
        model_dir = os.path.join('models', 'seed_'+str(seed))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_fn = os.path.join(model_dir, str(r.episode))
        r.agent.save_model(model_fn)
        print("Model saved to {}".format(model_fn))

        rew_dir = os.path.join('rew_figs', 'seed_'+str(seed))
        if not os.path.exists(rew_dir):
            os.makedirs(rew_dir)
        rew_fn = 'rew_' + str(r.episode) + '.png'
        rew_fn = os.path.join(rew_dir, rew_fn)
        plot_energy(r.episode_rewards, 'training episode', 'accumulated reward', rew_fn)
        energy_fn = 'final_energy_' + str(r.episode) + '.png'
        energy_fn = os.path.join(rew_dir, energy_fn)
        plot_energy(env.final_energy, 'training episode', 'final energy', energy_fn)
        energy_fn = 'ts_energy_' + str(r.episode) + '.png'
        energy_fn = os.path.join(rew_dir, energy_fn)
        plot_energy(env.ts_energy, 'training episode', 'ts energy', energy_fn)

    return True

runner.run(
        num_episodes=episodes,
        max_episode_timesteps=horizon,
        deterministic=False,
        episode_finished=episode_finished
    )
runner.close()