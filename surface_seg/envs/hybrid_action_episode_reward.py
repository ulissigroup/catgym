import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import os
import json
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.build import fcc111
from ase.visualize import view
from ase.calculators.emt import EMT as EMT_orig
from ase.constraints import FixAtoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from ase.visualize.plot import plot_atoms
from ase.io import write, read
from asap3 import EMT
from surface_seg.utils.countercalc import CounterCalc
from sella import Sella
import copy
from collections import OrderedDict
from .symmetry_function import make_snn_params, wrap_symmetry_functions
from sklearn.preprocessing import StandardScaler, Normalizer


ACTION_LOOKUP = [
    'move',
    'transition_state_search',
    'minimize_and_score',
#     'steepest_descent',
#     'steepest_ascent'
]

DIRECTION =[np.array([1,0,0]),
           np.array([-1,0,0]),
           np.array([0,1,0]),
           np.array([0,-1,0]),
           np.array([0,0,1]),
           np.array([0,0,-1]),
          ]

ELEMENT_LATTICE_CONSTANTS = {'Au':4.065 , 'Pd': 3.859, 'Ni': 3.499}


class MCSEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, size=(2, 2, 4),
                 element_choices={'Ni': 6, 'Pd': 5, 'Au': 5},
                 permute_seed=None,
                 save_dir=None,
                 step_size=0.1,
                 temperature = 1200,
                 observation_fingerprints = True,
                 observation_forces=True,
                 observation_positions = True,
                 descriptors = None,
                 timesteps = None,
                 thermal_threshold=None,
                 save_every = None,
                 save_every_min = None,
                 plot_every = None,
                 multi_env = True,
                 random_initial = False,
                 global_min_fps = None,
                 global_min_pos = None, 
                 Au_sublayer = False,
                 worse_initial = False,
                 random_free_atoms = None,
                 structure_idx = None,
                 different_initial=None,
                 
                ):
        self.different_initial = different_initial
        self.structure_idx = structure_idx
        self.random_free_atoms = random_free_atoms
        self.save_every_min = save_every_min
        self.worse_initial = worse_initial
        self.Au_sublayer = Au_sublayer
        self.random_initial = random_initial
        self.step_size = step_size
        self.thermal_threshold = thermal_threshold
        self.permute_seed = permute_seed
        self.size = size
        self.element_choices = element_choices
        self.descriptors = descriptors
        
        self.episodes = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.history_dir = os.path.join(save_dir, 'history')
        self.plot_dir = os.path.join(save_dir, 'plots')
        self.traj_dir = os.path.join(save_dir, 'trajs')
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.traj_dir):
            os.makedirs(self.traj_dir)
        
        self.save_every = save_every
        self.plot_every = plot_every
        self.multi_env = multi_env
        
        self.timesteps = timesteps
        self.initial_atoms, self.snn_params = self._get_initial_slab()
        self.initial_atoms.set_calculator(EMT())
            
        # Mark the free atoms
        self.free_atoms = list(set(range(len(self.initial_atoms))) -
                               set(self.initial_atoms.constraints[0].get_indices()))
        
        self.new_initial_atoms = self.initial_atoms.copy()
        self.initial_energy = self.initial_atoms.get_potential_energy()
        
        self.relative_energy = 0.0# self._get_relative_energy()
        self.initial_forces = self.initial_atoms.get_forces()[self.free_atoms]
        if not self.multi_env:
            self.minima_dir = os.path.join(save_dir, 'minima.traj')
            self.TS_dir = os.path.join(save_dir, 'TS.traj')
            write(self.minima_dir, self.initial_atoms) # save the initial config as a first local min.
            write(self.TS_dir, self.initial_atoms) # save the initial config as a first local min.

        
        self.atoms = self.initial_atoms.copy()
        
        self.observation_positions = observation_positions
        self.observation_fingerprints = observation_fingerprints
        
        self.observation_forces = observation_forces


        self.fps, self.fp_length = self._get_fingerprints(self.atoms)
        self.episode_initial_fps = self.fps.copy()
        self.positions = self.atoms.get_positions(wrap=False)[self.free_atoms]
#         self.initial_pos = self.pos.copy()
        
        self.total_force_calls = 0

        self.entire_unique_minima = [0.0]
        self.entire_unique_TS = [0.0]
        self.entire_unique_highest = [0.0]
        
        boltzmann_constant = 8.617e-5 #eV/K
        self.thermal_energy=temperature*boltzmann_constant*len(self.free_atoms)
        self.temperature = temperature
        # Define the possible actions

        self.action_space = spaces.Dict({'action_type': spaces.Discrete(4),
                                         'atom_selection': spaces.Discrete(8),
                                         'movement':spaces.Box(low=-self.step_size,
                                                               high=self.step_size,
                                                               shape=(1,3))})
        #Define the observation space
        self.observation_space = self._get_observation_space()
        
        # Set up the initial atoms
        self.reset()
        
        return

      # open AI gym API requirements
    def step(self, action):
        
        self.action_idx = action['action_type']
        self.steps = 50
        
        if self.action_idx == 0:
            self.atom_selection = action['atom_selection']
            self.movement = action['movement']
        else:
            self.atom_selection = 99
            self.movement = DIRECTION[4]
       
        self.previous_atoms = self.atoms.copy()
        previous_energy = self._get_relative_energy()

        self.done= False
        episode_over = False
        reward = 0
        
        if self.action_idx == 0:
            initial_positions = self.atoms.positions[np.array(self.free_atoms)[self.atom_selection],:].copy()
            self.atoms.positions[np.array(self.free_atoms)[self.atom_selection],:] = initial_positions + self.movement
        elif self.action_idx == 1:
            converged = self._transition_state_search()
        elif self.action_idx == 2:
            dyn = BFGSLineSearch(atoms=self.atoms, logfile=None)
            converged = dyn.run(0.03)
        elif self.action_idx == 3:    
            dyn = Langevin(self.atoms, 5 * units.fs, units.kB * self.temperature, 0.01, trajectory=None) # 5 fs --> 20 fs
            converged = dyn.run(self.steps) 

        #Add the reward for energy before/after 
        self.relative_energy = self._get_relative_energy()
        reward += self._get_reward(self.relative_energy, previous_energy)

        #Get the new observation
        observation = self._get_observation()
        
        
        if self.done:
            episode_over = True

        #Update the history for the rendering
        self.history, self.trajectories = self._update_history(self.action_idx, self.relative_energy)
        self.episode_reward += reward
        
        if len(self.history['actions'])-1 >= self.total_steps:
            episode_over = True
            
        if episode_over:
            self.total_force_calls += self.calc.force_calls
            self.min_idx = int(np.argmin(self.minima['energies']))

            if self.episodes % self.save_every == 0:
                self.save_episode()
                self.save_traj()
                
            if self.episodes % self.save_every_min == 0:
                if 1 in self.minima['segregation']:
                    self.save_episode()
                    self.save_traj()
                    
            self.episodes += 1
            
        return observation, reward, episode_over, {}
    
    def _get_new_initial_atoms(self):
        new_initial_atoms = self.atoms.copy()
        new_initial_atoms.set_calculator(self.calc)
        offset = self.atoms.get_scaled_positions(wrap=True)[self.free_atoms,:] - self.atoms.get_scaled_positions(wrap=False)[self.free_atoms,:]
        correction = offset @ self.atoms.get_cell().reshape(3,3)
        new_initial_atoms.positions[self.free_atoms,:] = self.atoms.positions[self.free_atoms,:] + correction
        
        assert np.abs(new_initial_atoms.get_potential_energy() - self.atoms.get_potential_energy()) < 1e-5
        return new_initial_atoms
    
    def _get_initial_slab(self):
        self.initial_atoms, self.elements = self._generate_slab(
            self.size, self.element_choices, self.permute_seed)
        if self.descriptors is None:
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
        self.snn_params = make_snn_params(self.elements, *descriptors)                
        return self.initial_atoms, self.snn_params
        
    
    def save_episode(self):
        save_path = os.path.join(self.history_dir, '%d_%f_%f_%f.npz' %(self.episodes, self.minima['energies'][self.min_idx],
                                                                   self.initial_energy, self.highest_energy))
        np.savez_compressed(save_path, 
                 initial_energy = self.initial_energy,
                 energies = self.history['energies'],
                 actions = self.history['actions'],
                 positions = self.history['positions'],
                 scaled_positions = self.history['scaled_positions'],
                 minima_energies = self.minima['energies'],
                 minima_steps = self.minima['timesteps'],
                 TS_energies = self.TS['energies'],
                 TS_steps = self.TS['timesteps'],
                 force_calls = self.history['force_calls'],
                 reward = self.episode_reward,
                 forces = self.history['forces']
                )
        if self.observation_fingerprints:
            np.savez_compressed(save_path, 
                 initial_energy = self.initial_energy,
                 energies = self.history['energies'],
                 actions = self.history['actions'],
#                  atom_selection = self.history['atom_selection'],
#                  movement = self.history['movement'],
#                  positions = self.history['positions'],
                 scaled_positions = self.history['scaled_positions'],
                 fingerprints = self.history['fingerprints'],
                initial_fps = self.history['initial_fps'],
#                  scaled_fingerprints = self.history['scaled_fingerprints'],
                 minima_energies = self.minima['energies'],
                 minima_steps = self.minima['timesteps'],
                minima_TS = self.minima['TS'],
                minima_highest_energyy = self.minima['highest_energy'],
                segregation = self.minima['segregation'],
#                  TS_energies = self.TS['energies'],
#                  TS_steps = self.TS['timesteps'],
                 force_calls = self.history['force_calls'],
                 total_force_calls = self.total_force_calls,
                 reward = self.episode_reward,
                 atomic_symbols = self.random_free_atoms,
                 structure_idx = [self.structure_idx],
#                  forces = self.history['forces']
                )
        return
    
    def save_traj(self):      
        save_path = os.path.join(self.traj_dir, '%d_%f_%f_%f_full.traj' %(self.episodes, self.minima['energies'][self.min_idx]
                                                                      , self.initial_energy, self.highest_energy))
        trajectories = []
        for atoms in self.trajectories:
            atoms.set_calculator(EMT())
            trajectories.append(atoms)
        write(save_path, trajectories)
        
        return
    
    def save_minima_TS(self):
        if self.multi_env:           
            if len(self.minima['trajectories']) > 1:
                minima = []
                for i, atoms in enumerate(self.minima['trajectories']):
                    atoms.set_calculator(EMT())
                    minima.append(atoms)
                write(os.path.join(self.traj_dir, '%d_%f_%f_%f_minima.traj' %(self.episodes, self.minima['energies'][self.min_idx], self.initial_energy, self.highest_energy)), minima)
               
        else:
            if len(self.minima['trajectories']) > 1: #exclude the initial state
                minima = read(self.minima_dir, index=':')
                for i, atoms in enumerate(self.minima['trajectories']):
                    if i != 0:
                        atoms.set_calculator(EMT())
                        minima.append(atoms)
                write(self.minima_dir, minima)
        return
    
    def plot_episode(self):
        save_path = os.path.join(self.plot_dir, '%d_%f_%f_%f.png' %(self.episodes, self.minima['energies'][self.min_idx]
                                                                , self.initial_energy, self.highest_energy))
            
        energies = np.array(self.history['energies'])
        actions = np.array(self.history['actions'])
        
        plt.figure(figsize=(9, 7.5))
        plt.xlabel('steps')
        plt.ylabel('energies')
        plt.plot(energies, color='black')
        
        for action_index in range(len(ACTION_LOOKUP)):
            action_time = np.where(actions==action_index)[0]
            plt.plot(action_time, energies[action_time], 'o', 
                    label=ACTION_LOOKUP[action_index])
        
        plt.scatter(self.minima['timesteps'], self.minima['energies'], label='minima', marker='x', color='r', s=180)
        plt.scatter(self.TS['timesteps'], self.TS['energies'], label='TS', marker='x', color='g', s=180)
        
        plt.legend(loc='upper left')
        plt.savefig(save_path, bbox_inches = 'tight')
        return plt.close('all')
    
    def evaluate_episode(self):    
        reward = 0
#         reward += max(self.thermal_threshold, self.highest_energy) * len(self.minima['energies'])
        
        return reward
    
    
    def reset(self):
        #Copy the initial atom and reset the calculator
        if os.path.exists('sella.log'):
            os.remove('sella.log')
            
        self.new_initial_atoms, self.snn_params = self._get_initial_slab()
        self.atoms = self.new_initial_atoms.copy()
#         self.snn_params = self._get_snn_params()
        self.atoms.set_calculator(EMT())
        _calc = self.atoms.get_calculator() # CounterCalc(EMT()) causes an error. This can bypass that error.
        self.calc = CounterCalc(_calc)
        self.atoms.set_calculator(self.calc)
#         self.initial_energy = self.atoms.get_potential_energy()
        self.highest_energy = 0.0
#         self.stables = [self._get_relative_energy()]
        self.bad_runs_min = 0
        self.bad_runs_TS = 0
        self.negative = False
        self.episode_reward = 0
        self.total_steps = self.timesteps
        self.max_height = 0
        
        #Set the list of identified positions
        self.minima = {}
        self.minima['positions'] = [self.atoms.positions[self.free_atoms,:].copy()]
        self.minima['energies'] = [0.0]
        self.minima['height'] = [0.0]
        self.minima['timesteps'] = [0]
        self.minima['trajectories'] = [self.atoms.copy()]
        self.minima['TS'] = [0.0]
        self.minima['highest_energy'] = [0.0]
        self.minima['segregation'] = []
        
        self.TS = {}
        self.TS['positions'] = []
        self.TS['energies'] = [0.0]
        self.TS['timesteps'] = []
        self.TS['trajectories'] = []

        self.move = {}
        self.move['trajectories'] = []

        #Set the energy history
        results = ['timesteps', 'energies','actions', 'positions', 'scaled_positions', 'fingerprints', 
                   'scaled_fingerprints', 'negative_energies', 'forces','atom_selection', 'movement', 'initial_fps']
        self.history = {}
        for item in results:
            self.history[item] = []
        self.history['timesteps'] = [0]
        self.history['energies'] = [0.0]
        self.history['actions'] = [2]
        self.history['force_calls'] = 0
        
        self.positions = self.atoms.get_scaled_positions(wrap=False)[self.free_atoms]
        self.initial_pos = self.positions
#         self.episode_initial_pos = self.initial_pos
        self.history['positions'] = [self.positions.tolist()]
        self.history['forces'] = [self.initial_forces.tolist()]
        self.history['scaled_positions'] = [self.atoms.get_scaled_positions(wrap=False)[self.free_atoms].tolist()]
        self.history['negative_energies'] = [0.0]
        self.history['atom_selection'] = [99]
        self.history['movement'] = [np.array([0,0,0]).tolist()]
#         self.history['moves'] = [np.zeros((8,3)).tolist()]
        if self.observation_fingerprints:
            self.fps, fp_length = self._get_fingerprints(self.atoms)
#             fps = fps#[self.free_atoms]
#             self.initial_fps = fps
            self.initial_fps = self.fps
            self.episode_initial_fps = self.fps
            self.history['fingerprints'] = [self.fps.tolist()]
            self.history['initial_fps'] = [self.episode_initial_fps.tolist()]
        self.trajectories = [self.atoms.copy()]        
        
#         self.num_calculations = []
        return self._get_observation()
    

    def render(self, mode='rgb_array'):

        if mode=='rgb_array':
            # return an rgb array representing the picture of the atoms
            
            #Plot the atoms
            fig, ax1 = plt.subplots()
            plot_atoms(self.atoms.repeat((3,3,1)), 
                       ax1, 
                       rotation='48x,-51y,-144z', 
                       show_unit_cell =0)
            
            ax1.set_ylim([0,25])
            ax1.set_xlim([-2, 20])
            ax1.axis('off')
            ax2 = fig.add_axes([0.35, 0.85, 0.3, 0.1])
            
            #Add a subplot for the energy history overlay           
            ax2.plot(self.history['timesteps'],
                     self.history['energies'])
            
            ax2.plot(self.minima['timesteps'],
                    self.minima['energies'],'o', color='r')
        
            if len(self.TS['timesteps']) > 0:
                ax2.plot(self.TS['timesteps'],
                        self.TS['energies'],'o', color='g')

            ax2.set_ylabel('Energy [eV]')
            
            #Render the canvas to rgb values for the gym render
            plt.draw()
            renderer = fig.canvas.get_renderer()
            x = renderer.buffer_rgba()
            img_array = np.frombuffer(x, np.uint8).reshape(x.shape)
            plt.close()
            
            #return the rendered array (but not the alpha channel)
            return img_array[:,:,:3]
            
        else:
            return
    
    def close(self):
        return
    
    def _update_history(self, action_idx, relative_energy):
        self.trajectories.append(self.atoms.copy())
        self.history['timesteps'] = self.history['timesteps'] + [self.history['timesteps'][-1] + 1]
        self.history['energies'] = self.history['energies'] + [self.relative_energy]
        self.history['actions'] = self.history['actions'] + [self.action_idx]
#         self.history['atom_selection'] = self.history['atom_selection'] + [self.atom_selection]
#         self.history['movement'] = self.history['movement'] + [self.movement]
        self.history['force_calls'] = self.calc.force_calls
        self.history['positions'] = self.history['positions'] + [self.atoms.get_positions(wrap=False)[self.free_atoms].tolist()]
#         self.history['forces'] = self.history['forces'] + [self.forces.tolist()]
        self.history['scaled_positions'] = self.history['scaled_positions'] + [self.atoms.get_scaled_positions(wrap=False)[self.free_atoms].tolist()]
        if self.observation_fingerprints:
            self.history['fingerprints'] = self.history['fingerprints'] + [self.fps.tolist()]
            self.history['initial_fps'] = self.history['initial_fps'] + [self.episode_initial_fps.tolist()]
        return self.history, self.trajectories
        

    
    def _get_reward(self, relative_energy, previous_energy):        
        reward = 0
        
        thermal_ratio=relative_energy/self.thermal_energy

        if relative_energy > self.highest_energy:
            self.highest_energy = relative_energy

        
        self.fps, self.fp_length = self._get_fingerprints(self.atoms)
        self.positions = self.atoms.get_scaled_positions(wrap=False)[self.free_atoms]
        
        relative_fps = self.fps - self.episode_initial_fps
    
        if np.max(np.array(self.positions)[:,2]) > self.max_height:
            self.max_height = np.max(np.array(self.positions)[:,2])
    
        if self.action_idx == 2: # Minimization
            minima_differences = np.abs(relative_energy-np.array(self.minima['energies']))
            
            if np.min(minima_differences) < 0.05:
                if self.highest_energy/self.thermal_energy > self.thermal_threshold / 2:
    #                 self.done = True
    #                 reward += 10 / self.highest_energy
#                     reward -= self.highest_energy - self.TS['energies'][-1]
#                     self.atoms.positions[self.free_atoms,:] = self.TS['positions'][-1]     
                    self.done = True

                else:
                    self.atoms.positions[self.free_atoms,:] = self.previous_atoms.positions[self.free_atoms,:].copy()

            elif np.min(minima_differences) > 0.05:
                if np.max(np.array(self.positions)[:,2]) < 0.66:
                    self.minima['energies'].append(relative_energy)
                    self.minima['timesteps'].append(self.history['timesteps'][-1] + 1)
                    self.minima['TS'].append(self.TS['energies'][-1])
                    self.minima['highest_energy'].append(self.highest_energy)

                if np.max(np.array(self.positions)[:,2]) < 0.66 and self.max_height > 0.66 and np.max(np.abs(relative_fps)) > 3:
                    reward += np.exp(-(relative_energy) / self.thermal_energy) / self.highest_energy
                    self.minima['segregation'].append(self.history['timesteps'][-1] + 1)
                    self.done = True
#                     self.highest_energy = relative_energy
#                     self.thermal_threshold += relative_energy/self.thermal_energy # adjusting threshold
#                     self.episode_initial_fps = self.fps.copy()
                    self.max_height = 0
        
        if self.action_idx == 1 and np.abs(relative_energy - previous_energy) > 0.01: #TS search
            TS_differences = np.abs(relative_energy-np.array(self.TS['energies']))
            if np.min(TS_differences) > 0.05 and relative_energy > np.max(self.TS['energies']):
#                 reward += relative_energy - np.max(self.TS['energies'])
                self.TS['energies'].append(relative_energy)
                self.TS['positions'].append(self.atoms.positions[self.free_atoms,:].copy())
                
        if thermal_ratio > self.thermal_threshold:
#             reward -= relative_energy + self.initial_energy #self.atoms.get_potential_energy()
            self.done = True
    
        return reward
    
    def sigmoid(self, thermal_ratio):
        L = 10
        k = 4
        x0 = self.thermal_threshold + 0
        return L / (1 + np.exp(-k * (thermal_ratio - x0)))

    def _transition_state_search(self):
        fix = self.atoms.constraints[0].get_indices()
        dyn = Sella(self.atoms,  # Your Atoms object
                    constraints=dict(fix=fix),  # Your constraints
#                     trajectory='saddle.traj',  # Optional trajectory,
                    logfile='sella.log'
                    )
        converged = dyn.run(0.05)#, steps = self.steps)#, steps=self.steps)
        return converged
    
    def _get_relative_energy(self):
        return self.atoms.get_potential_energy() - self.initial_energy

    def _get_observation(self):
        # helper function to get the current observation, which is just the position
        # of the free atoms as one long vector (should be improved)
           
#         observation = {'energy':np.array(self.relative_energy).reshape(1,)}
        observation = {}
        
        if self.observation_fingerprints:
            observation['fingerprints'] = self.fps - self.episode_initial_fps
            
        
        observation['positions'] = self.positions.flatten()
            
#         if self.observation_forces:
            
#             #Clip the forces to the state space limits to avoid really bad forces
#             forces = self.atoms.get_forces()[self.free_atoms, :]
           
#             forces = np.clip(forces, 
#                          self.observation_space['forces'].low[0],
#                          self.observation_space['forces'].high[0])
#         self.forces =  self.atoms.get_forces()[self.free_atoms, :]
#         observation['forces'] = self.forces.flatten()

        return observation
    
    def _get_fingerprints(self, atoms):
        #get fingerprints from amptorch as better state space feature
        fps = wrap_symmetry_functions(self.atoms, self.snn_params)
        fp_length = fps.shape[-1]
        
        return fps, fp_length
    
    def _get_observation_space(self):  
        
        observation_space = spaces.Dict({'fingerprints': spaces.Box(low=-6,
                                        high=6,
                                        shape=(len(self.atoms), self.fp_length)),
                                        'positions': spaces.Box(low=-1,
                                        high=2,
                                        shape=(len(self.free_atoms)*3,)),
#                                         'energy': spaces.Box(low=-2,
#                                                               high=2,
#                                                               shape=(1,)),
#                                          'forces': spaces.Box(low= -2,
#                                                              high= 2,
#                                                              shape=(len(self.free_atoms)*3,)
#                                                             )
                                        })

        return observation_space

    def _generate_slab(self, size, element_choices, permute_seed):
        # generate a pseudo-random sequence of elements
        
        if permute_seed is not None:
            np.random.seed(permute_seed)
            
        num_atoms = np.prod(size) # math.prod is only available in python 3.8
        atom_ordering = list(itertools.chain.from_iterable(
            [[key]*element_choices[key] for key in element_choices]))
        element_list = np.random.permutation(atom_ordering)
        # Use vergard's law to estimate the lattice constant
        a = np.sum([ELEMENT_LATTICE_CONSTANTS[key] *
                    element_choices[key]/num_atoms for key in element_choices])

        # Generate a base FCC slab
        slab = fcc111('Al', size=size, a=a, periodic=True, vacuum=10.0)
        slab.set_chemical_symbols(element_list)

        # Set the calculator
        slab.set_calculator(EMT())

        # Constrain the bottom two layers
        c = FixAtoms(indices=[atom.index for atom in slab if atom.position[2] < np.mean(
            slab.positions[:, 2])])  # Fix two layers
        slab.set_constraint(c)
        
        # reset the random seed for randomize initial configurations
        fixed_atoms_idx = c.get_indices()
        free_atoms_idx = list(set(np.arange(len(element_list))) ^ set(fixed_atoms_idx))
        free_atoms = element_list[free_atoms_idx]
        if self.random_initial:
            np.random.seed(None)
            self.random_free_atoms = np.random.permutation(free_atoms)
            new_element_list = list(element_list[fixed_atoms_idx]) + list(self.random_free_atoms)
            slab.set_chemical_symbols(new_element_list)        
        
        if self.different_initial:
            np.random.seed(None)
#             self.random_free_atoms = np.random.permutation(free_atoms)
            new_element_list = list(element_list[fixed_atoms_idx]) + list(self.random_free_atoms)
            slab.set_chemical_symbols(new_element_list)       
        
        if self.Au_sublayer:
            new_free_atoms = ['Au', 'Au', 'Au', 'Au', 'Ni', 'Pd', 'Pd', 'Ni'] # 4 Au in sublayer 
            new_element_list = list(element_list[fixed_atoms_idx]) + list(new_free_atoms)
            slab.set_chemical_symbols(new_element_list)  
            
        if self.worse_initial:
            new_free_atoms = ['Pd', 'Au', 'Pd', 'Au', 'Ni', 'Ni', 'Pd', 'Ni']
            new_element_list = list(element_list[fixed_atoms_idx]) + list(new_free_atoms)
            slab.set_chemical_symbols(new_element_list)  
                  
        # Do a quick minimization to relax the structure
        dyn = BFGSLineSearch(atoms=slab, logfile=None)
        dyn.run(0.03)
        elements = np.array(slab.symbols)
        _, idx = np.unique(elements, return_index=True)
        elements = list(elements[np.sort(idx)])
        
        return slab, elements
    
    
