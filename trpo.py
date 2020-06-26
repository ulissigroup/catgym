import gym
import gym.wrappers
import numpy as np
import tensorforce 
import copy
import tensorflow as tf

from surface_seg.envs.mcs_env import MCSEnv
from surface_seg.utils.callback import Callback


timesteps = 400

def setup_env(recording=True):
    
    # Set up gym
    MCS_gym = MCSEnv(observation_fingerprints=True, 
                     observation_forces=False,
                    permute_seed=42)
    
    if recording:
    # Wrap the gym to provide video rendering every 50 steps
        MCS_gym = gym.wrappers.Monitor(MCS_gym, 
                                         "./vid_trpo/fps", 
                                         force=True,
                                        video_callable = lambda episode_id: (episode_id)%50==0) #every 50, starting at 51
    
    #Convert gym to tensorforce environment
    env = tensorforce.environments.OpenAIGym(MCS_gym,
                                         max_episode_timesteps=timesteps,
                                         visualize=False)
    
    return env

# # Set up the gym and agent in tensorforce


from tensorforce.agents import Agent

agent = Agent.create(
    agent='trpo', 
    environment=setup_env(), 
    batch_size=50, 
    learning_rate=1e-3,
    memory = 40000,
    max_episode_timesteps = timesteps,
    exploration=dict(
        type='decaying', unit='timesteps', decay='exponential',
        initial_value=0.1, decay_steps=80000, decay_rate=0.5
    ),
    recorder = dict(
        directory = './recorder/fps', frequency=1), #required for recording states and actions
#     summarizer = dict(
#         directory = 'tb/fps', labels='all', frequency=, #Tensorboard summarizer
#     )
)
    

agent_spec = agent.spec


from tensorforce.execution import Runner
from surface_seg.utils.callback import Callback

#plot_frequency --> plotting energy and trajectories frequency
callback = Callback('./result_trpo/fps', plot_frequency=50).episode_finish

runner = Runner(
    agent=agent,
    environment=setup_env(recording=True),
    max_episode_timesteps=timesteps,
)

#callback_episode_frequency --> saving results and trajs frequency
runner.run(num_episodes=2000, callback=callback, callback_episode_frequency=1)
# runner.run(num_episodes=100, evaluation=True)
# runner.close()

# # Run the DRL method in parallel (multiple environments)


# from tensorforce.execution import Runner

# runner = Runner(
#     agent=agent_spec,
#     environments=[setup_env(), setup_env()],
#     num_parallel=2,
#     max_episode_timesteps=400,
# )

# runner.run(num_episodes=1000)
# runner.run(num_episodes=100, evaluation=True)
# runner.close()

