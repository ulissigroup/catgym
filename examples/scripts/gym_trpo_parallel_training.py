import matplotlib
matplotlib.use("Agg")
import gym
from surface_seg.envs.mcs_env import MCSEnv
import gym.wrappers
import numpy as np
import tensorforce 
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.execution import Runner
import os
import copy

def setup_env(recording=False):
    
    # Set up gym
    MCS_gym = MCSEnv(fingerprints=True, 
                    permute_seed=None)
    
    if recording:
    # Wrap the gym to provide video rendering every 50 steps
        MCS_gym = gym.wrappers.Monitor(MCS_gym, 
                                         "./vid", 
                                         force=True,
                                        video_callable = lambda episode_id: (episode_id)%50==0) #every 50, starting at 51
    
    #Convert gym to tensorforce environment
    env = tensorforce.environments.OpenAIGym(MCS_gym,
                                         max_episode_timesteps=400,
                                         visualize=False)
    
    return env

agent = Agent.create(
    agent='trpo', 
    environment=setup_env(), 
    batch_size=10, 
    learning_rate=1e-2,
    memory = 40000,
    max_episode_timesteps = 400,
    exploration=dict(
        type='decaying', unit='timesteps', decay='exponential',
        initial_value=0.3, decay_steps=1000, decay_rate=0.5
    ))

agent_spec = agent.spec

num_processes = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])

print('Detected N=%d cores, running in parallel!'%num_processes)

runner = Runner(
    agent=agent,
    environments=[setup_env() for _ in range(num_processes)],
    num_parallel=num_processes,
    remote='multiprocessing',
    max_episode_timesteps=400,
)

runner.run(num_episodes=2)
#runner.run(num_episodes=100, evaluation=True)
runner.close()
