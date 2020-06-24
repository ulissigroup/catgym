import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.build import fcc111
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.visualize.plot import plot_atoms
from asap3 import EMT
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

ELEMENT_LATTICE_CONSTANTS = {'Au':4.065 , 'Pd': 3.859, 'Ni': 3.499}


class MCSEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, size=(2, 2, 4),
                 element_choices={'Ni': 6, 'Pd': 5, 'Au': 5},
                 permute_seed=None,
                 step_size=0.4,
                 temperature = 1200,
                 observation_fingerprints = True,
                 observation_forces=True,
                 observation_positions = True,
                 descriptors = None):
            
        self.step_size = step_size
        
        self.initial_atoms, self.elements = self._generate_slab(
            size, element_choices, permute_seed)
        self.atoms = self.initial_atoms.copy()
        self.observation_positions = observation_positions
        self.observation_fingerprints = observation_fingerprints
        if self.observation_fingerprints:
            
            if descriptors is None:
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
            self.descriptors = descriptors
            self.snn_params = make_snn_params(self.elements, *descriptors)
            
        self.observation_forces = observation_forces
    
        # Mark the free atoms
        self.free_atoms = list(set(range(len(self.initial_atoms))) -
                               set(self.initial_atoms.constraints[0].get_indices()))

        boltzmann_constant = 8.617e-5 #eV/K
        self.thermal_energy=temperature*boltzmann_constant*len(self.free_atoms)
        
        # Define the possible actions
        self.action_space = spaces.Dict({'action_type':spaces.Discrete(len(ACTION_LOOKUP)),
                                         'atom_selection': spaces.Discrete(
                                                                      len(self.free_atoms)),
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
        self.action_idx = int(action['action_type'])
        action_type = ACTION_LOOKUP[self.action_idx]
        
#         self.trajectories.append(self.atoms.copy())
#         self.energies.append(self._get_relative_energy())
#         self.actions.append(self.action_idx)
#         self.positions.append(self.atoms.get_scaled_positions(wrap=False)[self.free_atoms].tolist())
        
        reward = 0
        if action_type == 'move':
            self._move_atom(action['atom_selection'], 
                            action['movement'])
            
        elif action_type == 'minimize_and_score':
            #This action tries to minimize
            reward+=self._minimize_and_score()

        elif action_type == 'transition_state_search':
            self._check_TS()

        elif action_type == 'steepest_descent':
            self._steepest_descent(action['atom_selection'])

        elif action_type == 'steepest_ascent':
            self._steepest_ascent(action['atom_selection'])

        else:
            raise Exception('I am not sure what action you mean!')

        #Get the new observation
        observation = self._get_observation()

        #Add the reward for energy before/after 
        relative_energy = self._get_relative_energy()
        reward += self._get_reward(relative_energy)

        #Update the history for the rendering
        self._update_history(self.action_idx, relative_energy)
        
        
#         self.atoms.wrap()
#         Stop if relative energy gets too high
#         if relative_energy > self.observation_space['energy'].high[0]:
#             episode_over = True
#         else:
#             episode_over = False
            
        episode_over=False

        return observation, reward, episode_over, {}

    def reset(self):
        #Copy the initial atom and reset the calculator
        self.atoms = self.initial_atoms.copy()
        self.atoms.set_calculator(EMT())
        self.initial_energy = self.atoms.get_potential_energy()
        self.highest_energy = 0.0
        
        #Set the list of identified positions
        self.minima = {}
        self.minima['positions'] = [self.atoms.positions[self.free_atoms,:].copy()]
        self.minima['energies'] = [self._get_relative_energy()]
        self.minima['timesteps'] = [0]
        
        self.TS = {}
        self.TS['positions'] = []
        self.TS['energies'] = []
        self.TS['timesteps'] = []
        
        #Set the energy history
        self.history = [(0, 2, 0.)] # (step, action, energy)
        self.num_calculations = []
        self.trajectories = [self.atoms.copy()]
        self.energies = [self._get_relative_energy()]
        self.actions = [2] # Initially minimized
        self.positions = [self.atoms.get_scaled_positions(wrap=False)[self.free_atoms].tolist()]
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
            history = np.array(self.history)
            ax2.plot(history[:,0],
                     history[:,2])
            
            ax2.plot(self.minima['timesteps'],
                    self.minima['energies'],'o')
            
            ax2.plot(self.TS['timesteps'],
                    self.TS['energies'],'o')
            
            #  ax2.set_xlim([0,200])
#             ax2.set_ylim([0,2])
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

    # Helper functions for above
    def _minimize(self):
        dyn = BFGSLineSearch(atoms=self.atoms, logfile=None)
        dyn.run(0.03)
        self.num_calculations.append((self.action_idx, dyn.get_number_of_steps()))
        return 
    
    def _update_history(self, action_idx, relative_energy):
        self.trajectories.append(self.atoms.copy())
        self.energies.append(self._get_relative_energy())
        self.actions.append(self.action_idx)
        self.positions.append(self.atoms.get_scaled_positions(wrap=False)[self.free_atoms].tolist())
        
        prev_step, prev_action, prev_energy = self.history[-1]
        self.history.append((prev_step + 1, action_idx, relative_energy))
        return
        

    def _minimize_and_score(self):
        reward = 0
        #Get the initial atom positions
        initial_positions = self.atoms.positions[self.free_atoms,:].copy()
        
        # Minimize and find the new position/energy
        self._minimize()
        current_positions = self.atoms.positions[self.free_atoms,:].copy()
        current_energy = self._get_relative_energy()
        
        #Get the distance from the new minima to every other found minima
        distances = [np.max(np.linalg.norm(current_positions-positions,1)) for positions in self.minima['positions']]
        energy_differences = np.abs(current_energy-np.array(self.minima['energies']))
        
        #If the distance is non-trivial, add it to the list and score it
        if np.min(distances)>0.2 or np.min(energy_differences)>0.1:
#             print('found a new local minima! distance=%1.2f w energy %1.2f'%(np.min(distances), current_energy))
            
            self.minima['positions'].append(current_positions)
            self.minima['energies'].append(current_energy)
            prev_step, prev_action, prev_energy = self.history[-1]
            self.minima['timesteps'].append(prev_step + 1)
            
#             reward=1000-current_energy*100+100*np.exp(-np.min(distances))
            reward = self.highest_energy - current_energy 
            self.highest_energy = current_energy
        #otherwise, reset the atoms positions since the minimization didn't do anything interesting
        else:
            self.atoms.positions[self.free_atoms,:]=initial_positions
#             reward = 0 
            # Penalizing the model when it triggered minimzation but not find new local minima (wasted time)
            reward -= current_energy
      
        return reward
    
    def _get_reward(self, relative_energy):        
        reward = 0
        
#         if relative_energy < 0:
            # we found a better minima! great!
    
        #Give rewards for moving, but reduce the reward for large energies
        thermal_ratio=relative_energy/self.thermal_energy
        if thermal_ratio>2:
            thermal_ratio = 2
        reward += 1-np.exp(thermal_ratio)
        
        #Add a reward based on the current TS
        if relative_energy > self.highest_energy:
            # we just went over a higher transition state! bad!
            reward += -(relative_energy - self.highest_energy)
            self.highest_energy = relative_energy
            
        return reward
    
    def _check_TS(self):
        reward = 0
        #Get the initial atom positions
        initial_positions = self.atoms.positions[self.free_atoms,:].copy()
        initial_energy = self._get_relative_energy()
        
        # Minimize and find the new position/energy
        self._transition_state_search()
        current_positions = self.atoms.positions[self.free_atoms,:].copy()
        current_energy = self._get_relative_energy()
        
        #Get the distance from the new minima to every other found minima
        if len(self.TS['energies'])==0:
            self._record_TS()
            
        else:
            distances = [np.max(np.linalg.norm(current_positions-positions,1)) for positions in self.TS['positions']]
            energy_differences = np.abs(current_energy-np.array(self.TS['energies']))

            #If the distance is non-trivial, add it to the list and score it
            if np.min(distances)>0.2 or np.min(energy_differences)>0.1:
                self._record_TS()

            else:
                # Penalizing the model when it triggered Ts search but not find new TS(wasted time)
                reward -= current_energy
                self.atoms.positions[self.free_atoms,:]=initial_positions
        return
    
    def _record_TS(self):
        self.TS['positions'].append(self.atoms.positions[self.free_atoms,:].copy())
        self.TS['energies'].append(self._get_relative_energy())
            
        prev_step, prev_action, prev_energy = self.history[-1]
        self.TS['timesteps'].append(prev_step + 1)
        
    def _transition_state_search(self):
        fix = self.atoms.constraints[0].get_indices()
        dyn = Sella(self.atoms,  # Your Atoms object
                    constraints=dict(fix=fix),  # Your constraints
#                     trajectory='saddle.traj',  # Optional trajectory,
                    logfile='sella.log'
                    )
        dyn.run(1e-2, steps=10)
        self.num_calculations.append((self.action_idx, dyn.get_number_of_steps()))
        return 
    
    def _get_relative_energy(self):
        return self.atoms.get_potential_energy() - self.initial_energy

    def _steepest_descent(self, atom_index):
        force = self.atoms.get_forces()[self.free_atoms[atom_index], :]
        move = -self.step_size*force
        self._movement_line_search(atom_index, move)
        return

    def _steepest_ascent(self, atom_index):
        force = self.atoms.get_forces()[self.free_atoms[atom_index], :]
        move = self.step_size*force
        self._movement_line_search(atom_index, move)
        return

    def _move_atom(self, atom_index, movement):
        # Helper function to move an atom
        self._movement_line_search(atom_index, 
                                   movement)
        return
    
    def _movement_line_search(self, atom_index, movement, reasonable_step_energy = 0.5):
        #like move_atom, but do a line search so that we don't take too large of a step
        
        #Get the initial position/energy of the atom in question
        initial_energy = self._get_relative_energy()
        initial_position = self.atoms.positions[self.free_atoms[atom_index],:].copy()
                                                 
        #Make sure the specified movement is a 3x1 vector
        movement = movement.reshape((3,))
        
        #Get the force on the atom
        atom_force = self.atoms.get_forces()[self.free_atoms[atom_index],:]
        
        # estimate the energy associated with the trial move
        estimated_move_energy = np.abs(np.dot(atom_force,movement))
        
        #scale the movement, assuming the PES is linear at the initial point. 
        # could be improved with a quadratic approximation.
        # or a call to scipy.optimize using the force at each point in the obj function
        if estimated_move_energy>reasonable_step_energy:
            trial_movement = movement*reasonable_step_energy/estimated_move_energy
        else:
            trial_movement = movement
        
        #When making the move, scale the step down so that it is actually less than the reasonable
        # step energy specified above
        
        trial_movement_energy = 1.0
        while trial_movement_energy>reasonable_step_energy:
            self.atoms.positions[self.free_atoms[atom_index],:] = initial_position + trial_movement
            
            #reduce the trial_movement by a factor of 2 to find a more reasonable step
            trial_movement_energy = np.abs(self._get_relative_energy()-initial_energy)
            trial_movement /= 2
            
        return

    def _get_observation(self):
        # helper function to get the current observation, which is just the position
        # of the free atoms as one long vector (should be improved)
           
        #Clip the energy to the state space limits to avoid really bad energies
        relative_energy = self._get_relative_energy()
#         if relative_energy > self.observation_space['energy'].high[0]:
#             relative_energy = self.observation_space['energy'].high[0]
        
#         observation = {'energy':np.array([relative_energy])}
        observation = {}
        
        if self.observation_fingerprints:
            fps, fp_length = self._get_fingerprints(self.atoms)
            fps = fps[self.free_atoms]
            fps = StandardScaler().fit_transform(fps)
            fps = Normalizer().fit_transform(fps)
            fps = fps.flatten()
            observation['fingerprints'] = fps#.reshape(-1, fp_length)
#             observation['fingerprints'] = fps[self.free_atoms]
            
        observation['positions'] = self.atoms.get_scaled_positions(wrap=False)[self.free_atoms].flatten()
            
        if self.observation_forces:
            
            #Clip the forces to the state space limits to avoid really bad forces
            forces = self.atoms.get_forces()[self.free_atoms, :]
        
            forces = np.clip(forces, 
                         self.observation_space['forces'].low[0,0],
                         self.observation_space['forces'].high[0,0])
        
            observation['forces'] = forces

        return observation
    
    def _get_fingerprints(self, atoms):
        #get fingerprints from amptorch as better state space feature
        fps = wrap_symmetry_functions(self.atoms, self.snn_params)
        fp_length = fps.shape[-1]
        
        return fps, fp_length
    
    def _get_observation_space(self):  
        # Can make more space options later.
        # Flattened atomic positions and fingerprints to not use the default conv1d layers.
        # Otherwise, the policy networks consider these states as images and use standard filter to pool these vectors, which does not make sense.
        # For now, no lower/upper bounds
        # Scaled and Normarlized fingerprints for better updating weights in the policy networks
        # Dit not include energies because (1,) input is too short for network training? 
        # The default (auto) network we are now using will expand (1,) --> (64,). Not sure if it can learn meaningful observations by doing it.
        
        if self.observation_fingerprints and self.observation_positions:
            fps, fp_length = self._get_fingerprints(self.atoms)
            observation_space = spaces.Dict({'fingerprints': spaces.Box(low=np.inf,
                                            high=np.inf,
                                            shape=(len(self.free_atoms)*fp_length, )),
                                            'positions': spaces.Box(low=-1,
                                            high=2,
                                            shape=(len(self.free_atoms)*3,))})

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

        # Do a quick minimization to relax the structure
        dyn = BFGSLineSearch(atoms=slab, logfile=None)
        dyn.run(0.1)
        
        elements = np.array(slab.symbols)
        _, idx = np.unique(elements, return_index=True)
        elements = list(elements[np.sort(idx)])
        
        return slab, elements

