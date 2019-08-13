from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import json
import logging
import os
import time
import sys
import numpy as np

from matplotlib import pyplot as plt
from tensorforce import TensorForceError
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
# from tensorforce.contrib.openai_gym import OpenAIGym

from surface_env import *


def plot_energy(energy, ylabel, save_path):
    plt.figure()
    plt.xlabel('training episode')
    plt.ylabel(ylabel)
    plt.title('episode vs. ' + ylabel)
    plt.plot(energy)
    plt.savefig(save_path)
    print('figure saved as {}'.format(save_path))
    return 


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--episodes', type=int, default=int(1e5), help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-hrz', '--horizon', type=int, default=800, help="Horizon of each episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=50, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")

    args = parser.parse_args()

    env = SurfaceEnv(args.horizon)
    print('Initial energy:', env.get_energy())
    
    # lattice = Surface()
    # lattice.reset()
    # print(lattice.current_positions())
    # print(lattice.current_positions().shape)
    # print(type(lattice.current_positions()))

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
        # {
        #     "type": "dense",
        #     "size": 6+32,
        #     "activation": "softmax"
        # }
    ]

    # print(env.states)
    # print(env.states['shape'])

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

    def episode_finished(r):
        # positions = env.get_positions()
        # print("pos diff:", np.sum(positions[1,:] - positions[0,:]))
        # np.save('shit', positions)
        if r.episode % 50 == 0:
            positions = env.get_positions()
            pos_fn = '_'.join(['pos_3', str(r.episode)])
            pos_dir = os.path.join('new_pos', pos_fn)
            np.save(pos_dir, positions)
        if r.episode % 50 == 0:
            agent_fn = '_'.join(['agent_3', str(r.episode)])
            agent_dir = os.path.join('new_agents', agent_fn)
            r.agent.save_model(agent_dir)
            print("Saving agent to {}".format(agent_dir))
        if r.episode % 50 == 0:
            rew_fn = '.'.join(['_'.join(['reward_3', str(r.episode)]), 'png'])
            rew_dir = os.path.join('new_plots', rew_fn)
            plot_energy(r.episode_rewards, 'accumulated reward', rew_dir)
            energy_fn = '.'.join(['_'.join(['final', 'energy_3', str(r.episode)]), 'png'])
            energy_dir = os.path.join('new_plots', energy_fn)
            plot_energy(env.final_energy, 'final energy', energy_dir)
        print("Finished episode {ep} after {ts} timesteps (reward: {reward})".
            format(ep=r.episode, ts=r.episode_timestep,reward=r.episode_rewards[-1]))
        return True

    runner.run(
        # num_timesteps=args.timesteps,
        num_episodes=args.episodes,
        max_episode_timesteps=args.horizon,
        deterministic=args.deterministic,
        episode_finished=episode_finished
    )
    runner.close()


if __name__ == '__main__':
    main()
