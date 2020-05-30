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

def setup_env(recording=True):
    
    Gs = {}
    Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
    Gs["G2_rs_s"] = [0] * 4
    Gs["G4_etas"] = [0.005]
    Gs["G4_zetas"] = [1.0]
    Gs["G4_gammas"] = [+1.0, -1]
    Gs["cutoff"] = 6.5

    G = copy.deepcopy(Gs)

    # order descriptors for simple_nn
    cutoff = G["cutoff"]
    G["G2_etas"] = [a / cutoff**2 for a in G["G2_etas"]]
    G["G4_etas"] = [a / cutoff**2 for a in G["G4_etas"]]
    descriptors = (
        G["G2_etas"],
        G["G2_rs_s"],
        G["G4_etas"],
        G["cutoff"],
        G["G4_zetas"],
        G["G4_gammas"],
    )

    # Set up gym
    MCS_gym = MCSEnv(fingerprints=True, 
                     descriptors = descriptors,
                    permute_seed=None)
    
    if recording:
    # Wrap the gym to provide video rendering every 50 steps
        MCS_gym = gym.wrappers.Monitor(MCS_gym, 
                                         "./vid", 
                                         force=True,
                                        video_callable = lambda episode_id: (episode_id)%50==0)
    
    #Convert gym to tensorfce environment
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
    agent=agent_spec,
    environments=[setup_env() for _ in range(num_processes)],
    num_parallel=num_processes,
    remote='multiprocessing',
    max_episode_timesteps=400,
)

runner.run(num_episodes=1000)
#runner.run(num_episodes=100, evaluation=True)
runner.close()
