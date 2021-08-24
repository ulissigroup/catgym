import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import os
import json
import itertools
import matplotlib

matplotlib.use("agg")
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

from surface_seg.utils.traindataloader import read_images
from amptorch.trainer import AtomsTrainer
from amptorch import AMPtorch
import torch
from al_mlp.ml_potentials.flare_pp_calc import FlarePPCalc
from al_mlp.online_learner.online_learner import OnlineLearner


ACTION_LOOKUP = ["Move", "TS", "Min", "MD"]

DIRECTION = [
    np.array([1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, -1, 0]),
    np.array([0, 0, 1]),
    np.array([0, 0, -1]),
]

ELEMENT_LATTICE_CONSTANTS = {"Au": 4.065, "Pd": 3.859, "Ni": 3.499}


class MCSEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        size=(2, 2, 4),
        element_choices={"Ni": 6, "Pd": 5, "Au": 5},
        permute_seed=None,
        save_dir=None,
        step_size=0.1,
        temperature=1200,
        observation_fingerprints=True,
        observation_forces=True,
        observation_positions=True,
        descriptors=None,
        timesteps=None,
        thermal_threshold=None,
        save_every=None,
        save_every_min=None,
        random_initial=False,
        global_min_fps=None,
        global_min_pos=None,
        Au_sublayer=False,
        worse_initial=False,
        random_free_atoms=None,
        structure_idx=None,
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
        self.history_dir = os.path.join(save_dir, "history")
        self.plot_dir = os.path.join(save_dir, "plots")
        self.traj_dir = os.path.join(save_dir, "trajs")
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.traj_dir):
            os.makedirs(self.traj_dir)

        self.save_every = save_every

        self.timesteps = timesteps
        self.initial_atoms, self.snn_params = self._get_initial_slab()
        self.initial_atoms.set_calculator(self.ML_calc)
        self.initial_positions = self.initial_atoms.get_positions()

        # Mark the free atoms
        self.free_atoms = list(
            set(range(len(self.initial_atoms)))
            - set(self.initial_atoms.constraints[0].get_indices())
        )

        self.new_initial_atoms = self.initial_atoms.copy()
        self.initial_energy = self.initial_atoms.get_potential_energy()
        self.initial_forces = self.initial_atoms.get_forces()[self.free_atoms]
        self.relative_energy = 0.0  # self._get_relative_energy()

        self.atoms = self.initial_atoms.copy()

        self.observation_positions = observation_positions
        self.observation_fingerprints = observation_fingerprints
        self.observation_forces = observation_forces

        self.fps, self.fp_length = self._get_fingerprints(self.atoms)
        self.episode_initial_fps = self.fps.copy()
        self.positions = self.atoms.get_positions(wrap=False)[self.free_atoms]
        #         self.initial_pos = self.pos.copy()

        self.total_force_calls = 0

        boltzmann_constant = 8.617e-5  # eV/K
        self.thermal_energy = temperature * boltzmann_constant * len(self.free_atoms)
        self.temperature = temperature
        # Define the possible actions

        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(4),
                "atom_selection": spaces.Discrete(8),
                "movement": spaces.Box(
                    low=-self.step_size, high=self.step_size, shape=(1, 3)
                ),
            }
        )
        # Define the observation space
        self.observation_space = self._get_observation_space()

        # Set up the initial atoms
        self.reset()

        return

    # Initialize the machine learning potential and online calculator
    # Parameters hardcoded for now
    # Parent calculator is EMT, can be changed to Vasp/VaspInteractive
    def initialize_MLcalc(self, initial_slab):
        learner_params = {
            "stat_uncertain_tol": 0.08,
            "dyn_uncertain_tol": 0.1,
            # make sure fmax_verify_threshold is consistent with fmax in minimization
            "fmax_verify_threshold": 0.05,
        }

        flare_params = {
            "sigma": 2,
            "power": 2,
            "cutoff_function": "quadratic",
            "cutoff": 5.0,
            "radial_basis": "chebyshev",
            "cutoff_hyps": [],
            "sigma_e": 0.002,
            "sigma_f": 0.05,
            "sigma_s": 0.0006,
            "max_iterations": 50,
            "freeze_hyps": 0,
        }

        ml_potential = FlarePPCalc(flare_params, [initial_slab])
        self.ML_calc = OnlineLearner(
            learner_params,
            [],
            ml_potential,
            EMT(),  # parent calculator
        )

    # open AI gym API requirements
    def step(self, action):
        reward = 0

        self.action_idx = action["action_type"]
        self.steps = 50

        self.previous_atoms = self.atoms.copy()
        previous_energy = self._get_relative_energy()
        previous_force = np.max(np.abs(self.atoms.get_forces()[8:]))
        self.done = False
        episode_over = False

        save_path_min = None
        save_path_ts = None
        save_path_md = None

        if ACTION_LOOKUP[self.action_idx] == "Move":
            self.atom_selection = action["atom_selection"]
            self.movement = action["movement"]
            initial_positions = self.atoms.positions[
                np.array(self.free_atoms)[self.atom_selection], :
            ].copy()
            self.atoms.positions[np.array(self.free_atoms)[self.atom_selection], :] = (
                initial_positions + self.movement
            )

        elif ACTION_LOOKUP[self.action_idx] == "TS":
            #             dyn = self._transition_state_search()
            fix = self.atoms.constraints[0].get_indices()
            self.dyn_TS = Sella(
                self.atoms,  # Your Atoms object
                constraints=dict(fix=fix),  # Your constraints
                trajectory=save_path_ts,  # Optional trajectory,
                logfile="sella.log",
            )
            converged = self.dyn_TS.run(0.05, steps=self.steps)
            self.TS_atoms = self.atoms.copy()

        elif ACTION_LOOKUP[self.action_idx] == "Min":
            if self.found_TS == 1:
                self.H = self.TS_H.copy()
                eig, V = np.linalg.eigh(self.H)
                a = (self.initial_positions - self.atoms.positions).reshape(-1)
                b = V[:, 0]
                direction = np.dot(a, b)
                angle = math.acos(direction / np.linalg.norm(a) / np.linalg.norm(b))
                # Move away from the initial point
                #                 dr =  direction * V[:,0].reshape(-1,3)
                dr = np.sign(direction) * V[:, 0].reshape(-1, 3)
                self.atoms.set_positions(self.TS["positions"][-1] - dr)
            #                 self.atoms.set_positions(self.atoms.get_positions() - dr)

            dyn = BFGSLineSearch(
                atoms=self.atoms, logfile=None, trajectory=save_path_min
            )
            converged = dyn.run(0.03)

        elif ACTION_LOOKUP[self.action_idx] == "MD":
            dyn = Langevin(
                self.atoms,
                5 * units.fs,
                units.kB * self.temperature,
                0.01,
                trajectory=save_path_md,
            )  # 5 fs --> 20
            converged = dyn.run(self.steps)

        # Add the reward for energy before/after
        self.relative_energy = self._get_relative_energy()
        reward += self._get_reward(self.relative_energy, previous_energy)

        # Get the new observation
        observation = self._get_observation()

        if self.done:
            episode_over = True

        # Update the history for the rendering
        self.relative_energy = self._get_relative_energy()
        self.history, self.trajectories = self._update_history(
            self.action_idx, self.relative_energy
        )
        self.episode_reward += reward

        if len(self.history["actions"]) - 1 >= self.total_steps:
            episode_over = True

        if episode_over:
            self.total_force_calls += self.calc.force_calls
            self.min_idx = int(np.argmin(self.minima["energies"]))
            # Save Episode
            if self.episodes % self.save_every == 0:
                self.save_episode()
                self.save_traj()
            # Save Episode if it finds segregation
            if self.episodes % self.save_every_min == 0:
                if len(self.minima["segregation"]) > 0:
                    self.save_episode()
                    self.save_traj()

            self.episodes += 1

        return observation, reward, episode_over, {}

    def reset(self):

        # Copy the initial atom and reset the calculator
        if os.path.exists("sella.log"):
            os.remove("sella.log")
        # Initialize Atoms and Calculator
        self.new_initial_atoms, self.snn_params = self._get_initial_slab()
        self.atoms = self.new_initial_atoms.copy()
        self.atoms.set_calculator(self.ML_calc)
        _calc = self.atoms.get_calculator()
        self.calc = CounterCalc(_calc)
        self.atoms.set_calculator(self.calc)

        self.highest_energy = 0.0
        self.episode_reward = 0
        self.total_steps = self.timesteps
        self.max_height = 0
        self.action_idx = 0

        # Set the list of identified positions
        self.minima = {}
        self.minima["positions"] = [self.atoms.positions.copy()]
        self.minima["energies"] = [0.0]
        self.minima["height"] = [0.0]
        self.minima["timesteps"] = [0]
        self.minima["trajectories"] = [self.atoms.copy()]
        self.minima["TS"] = [0.0]
        self.minima["highest_energy"] = [0.0]
        self.minima["segregation"] = []

        self.TS = {}
        self.TS["positions"] = [self.atoms.positions[self.free_atoms, :].copy()]
        self.TS["energies"] = [0.0]
        self.TS["timesteps"] = []
        self.TS["trajectories"] = []

        self.found_TS = 0

        self.min_traj = [self.atoms.copy()]
        self.min_energies = [0]
        self.ts_traj = [self.atoms.copy()]
        self.ts_energies = [0]
        self.md_traj = [self.atoms.copy()]
        self.md_energies = [0]

        self.move = {}
        self.move["trajectories"] = []

        # Set the energy history
        results = [
            "timesteps",
            "energies",
            "actions",
            "positions",
            "scaled_positions",
            "fingerprints",
            "scaled_fingerprints",
            "negative_energies",
            "forces",
            "atom_selection",
            "movement",
            "initial_fps",
        ]
        self.history = {}
        for item in results:
            self.history[item] = []
        self.history["timesteps"] = [0]
        self.history["energies"] = [0.0]
        self.history["actions"] = [2]
        self.history["force_calls"] = 0

        self.positions = self.atoms.get_scaled_positions(wrap=False)[self.free_atoms]
        self.initial_pos = self.positions
        self.history["positions"] = [self.positions.tolist()]
        self.history["forces"] = [self.initial_forces.tolist()]
        self.history["scaled_positions"] = [
            self.atoms.get_scaled_positions(wrap=False)[self.free_atoms].tolist()
        ]
        self.history["negative_energies"] = [0.0]
        self.history["atom_selection"] = [99]
        self.history["movement"] = [np.array([0, 0, 0]).tolist()]
        if self.observation_fingerprints:
            self.fps, fp_length = self._get_fingerprints(self.atoms)
            self.initial_fps = self.fps
            self.episode_initial_fps = self.fps
            self.history["fingerprints"] = [self.fps.tolist()]
            self.history["initial_fps"] = [self.episode_initial_fps.tolist()]
        self.trajectories = [self.atoms.copy()]

        #         self.num_calculations = []
        return self._get_observation()

    def _get_reward(self, relative_energy, previous_energy):
        reward = 0

        thermal_ratio = relative_energy / self.thermal_energy

        if relative_energy > self.highest_energy:
            self.highest_energy = relative_energy

        if thermal_ratio > self.thermal_threshold:
            reward -= thermal_ratio
            self.done = True

        self.fps, self.fp_length = self._get_fingerprints(self.atoms)
        self.orig_positions = self.atoms.get_positions().copy()
        self.positions = self.atoms.get_scaled_positions(wrap=False)[self.free_atoms]
        relative_fps = self.fps - self.episode_initial_fps

        if np.max(np.array(self.positions)[:, 2]) > self.max_height:
            self.max_height = np.max(np.array(self.positions)[:, 2])

        if ACTION_LOOKUP[self.action_idx] == "TS":
            TS_differences = np.abs(relative_energy - np.array(self.TS["energies"]))
            if np.min(TS_differences) < 0.05:
                # Action is Rejected
                self.atoms.positions[
                    self.free_atoms, :
                ] = self.previous_atoms.positions[self.free_atoms, :].copy()
                self.relative_energy = previous_energy

            if relative_energy > 1.5 and np.max(np.array(self.positions)[:, 2]) > 0.68:
                if np.min(TS_differences) > 0.1:  # New TS based on E
                    if self.found_TS == 0:  # If no TS has been previously found
                        self.TS_H = self.dyn_TS.pes.H.asarray().copy()
                        eig, V = np.linalg.eigh(self.TS_H)
                        if len(eig[eig < 0]) == 1:  # Check first order TS
                            self.found_TS = 1
                            reward += 1 / relative_energy
                            #                             reward += 0.5
                            self._save_TS(relative_energy)
                    elif self.found_TS == 1:
                        eig, V = np.linalg.eigh(self.dyn_TS.pes.H.asarray())
                        if len(eig[eig < 0]) == 1:
                            self._save_TS(relative_energy)
                            self.TS_H = self.dyn_TS.pes.H.asarray().copy()

        if ACTION_LOOKUP[self.action_idx] == "Min":
            minima_differences = np.abs(
                self._get_relative_energy() - np.array(self.minima["energies"])
            )
            if self.found_TS == 1:
                if (
                    np.max(np.array(self.positions)[:, 2]) < 0.66
                    and np.max(np.abs(relative_fps)) > 3
                    and np.min(minima_differences) > 0.05
                ):  # No additional layer allowed
                    self._save_minima()
                    reward += 2 * np.exp(-thermal_ratio) / self.highest_energy
                    self.minima["segregation"].append(self.history["timesteps"][-1] + 1)
                    self.done = True  # Terminate the episode
                else:
                    self.atoms.set_positions(self.TS["positions"][-1])
                    self.relative_energy = self.TS["energies"][-1]

            else:  # fonud_TS == 0
                if np.min(minima_differences) < 0.05:
                    # Action is Rejected
                    self.atoms.positions[
                        self.free_atoms, :
                    ] = self.previous_atoms.positions[self.free_atoms, :].copy()
                    self.relative_energy = previous_energy
                else:
                    self._save_minima()

        return reward

    def _get_observation(self):
        # helper function to get the current observation, which is just the position
        # of the free atoms as one long vector (should be improved)

        observation = {
            "energy": np.array(self._get_relative_energy()).reshape(
                1,
            )
        }
        #         observation = {}
        if self.observation_fingerprints:
            observation["fingerprints"] = (self.fps - self.episode_initial_fps)[
                self.free_atoms, :
            ].flatten()

        observation["positions"] = self.atoms.get_scaled_positions()[
            self.free_atoms, :
        ].flatten()

        if self.observation_forces:
            self.forces = self.atoms.get_forces()[self.free_atoms, :]
            observation["forces"] = self.forces.flatten()

        observation["TS"] = np.array([self.found_TS]).reshape(
            1,
        )
        return observation

    def _get_observation_space(self):
        observation_space = spaces.Dict(
            {
                "fingerprints": spaces.Box(
                    low=-6, high=6, shape=(len(self.free_atoms) * self.fp_length,)
                ),
                "positions": spaces.Box(
                    low=-1, high=2, shape=(len(self.free_atoms) * 3,)
                ),
                "energy": spaces.Box(low=-1, high=2.5, shape=(1,)),
                "forces": spaces.Box(low=-2, high=2, shape=(len(self.free_atoms) * 3,)),
                "TS": spaces.Box(low=-0.5, high=1.5, shape=(1,)),
            }
        )

        return observation_space

    def _save_minima(self):
        self.minima["energies"].append(self._get_relative_energy())
        self.minima["timesteps"].append(self.history["timesteps"][-1] + 1)
        self.minima["TS"].append(self.TS["energies"][-1])
        self.minima["highest_energy"].append(self.highest_energy)
        self.minima["positions"].append(self.atoms.positions.copy())
        return

    def _save_TS(self, relative_energy):
        self.TS["energies"].append(relative_energy)
        self.TS["positions"].append(self.atoms.get_positions().copy())
        return

    def _transition_state_search(self):
        fix = self.atoms.constraints[0].get_indices()
        dyn = Sella(
            self.atoms,  # Your Atoms object
            constraints=dict(fix=fix),  # Your constraints
            #                     trajectory='saddle.traj',  # Optional trajectory,
            logfile="sella.log",
        )
        converged = dyn.run(0.05)  # , steps = self.steps)#, steps=self.steps)
        return dyn

    def _get_relative_energy(self):
        return self.atoms.get_potential_energy() - self.initial_energy

    def _get_fingerprints(self, atoms):
        # get fingerprints from amptorch as better state space feature
        fps = wrap_symmetry_functions(self.atoms, self.snn_params)
        fp_length = fps.shape[-1]

        return fps, fp_length

    def _get_initial_slab(self):
        self.initial_atoms, self.elements = self._generate_slab(
            self.size, self.element_choices, self.permute_seed
        )
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
            G["G2_etas"] = [a / cutoff ** 2 for a in G["G2_etas"]]
            G["G4_etas"] = [a / cutoff ** 2 for a in G["G4_etas"]]
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

    def _generate_slab(self, size, element_choices, permute_seed):
        # generate a pseudo-random sequence of elements

        if permute_seed is not None:
            np.random.seed(permute_seed)

        num_atoms = np.prod(size)  # math.prod is only available in python 3.8
        atom_ordering = list(
            itertools.chain.from_iterable(
                [[key] * element_choices[key] for key in element_choices]
            )
        )
        element_list = np.random.permutation(atom_ordering)
        # Use vergard's law to estimate the lattice constant
        a = np.sum(
            [
                ELEMENT_LATTICE_CONSTANTS[key] * element_choices[key] / num_atoms
                for key in element_choices
            ]
        )

        # Generate a base FCC slab
        slab = fcc111("Al", size=size, a=a, periodic=True, vacuum=10.0)
        slab.set_chemical_symbols(element_list)

        # Constrain the bottom two layers
        c = FixAtoms(
            indices=[
                atom.index
                for atom in slab
                if atom.position[2] < np.mean(slab.positions[:, 2])
            ]
        )  # Fix two layers
        slab.set_constraint(c)

        # reset the random seed for randomize initial configurations
        fixed_atoms_idx = c.get_indices()
        free_atoms_idx = list(set(np.arange(len(element_list))) ^ set(fixed_atoms_idx))
        free_atoms = element_list[free_atoms_idx]
        if self.random_initial:
            np.random.seed(None)
            self.random_free_atoms = np.random.permutation(free_atoms)
            new_element_list = list(element_list[fixed_atoms_idx]) + list(
                self.random_free_atoms
            )
            slab.set_chemical_symbols(new_element_list)

        if self.different_initial:
            np.random.seed(None)
            #             self.random_free_atoms = np.random.permutation(free_atoms)
            new_element_list = list(element_list[fixed_atoms_idx]) + list(
                self.random_free_atoms
            )
            slab.set_chemical_symbols(new_element_list)

        if self.Au_sublayer:
            new_free_atoms = [
                "Au",
                "Au",
                "Au",
                "Au",
                "Ni",
                "Pd",
                "Pd",
                "Ni",
            ]  # 4 Au in sublayer
            new_element_list = list(element_list[fixed_atoms_idx]) + list(
                new_free_atoms
            )
            slab.set_chemical_symbols(new_element_list)

        if self.worse_initial:
            new_free_atoms = ["Pd", "Au", "Pd", "Au", "Ni", "Ni", "Pd", "Ni"]
            new_element_list = list(element_list[fixed_atoms_idx]) + list(
                new_free_atoms
            )
            slab.set_chemical_symbols(new_element_list)

        # Set the calculator
        # initialize ML calc at 0th episode
        if self.episodes == 0:
            self.initialize_MLcalc(slab)
        slab.set_calculator(self.ML_calc)

        # Do a quick minimization to relax the structure
        dyn = BFGSLineSearch(atoms=slab, logfile=None)
        dyn.run(0.03)
        elements = np.array(slab.symbols)
        _, idx = np.unique(elements, return_index=True)
        elements = list(elements[np.sort(idx)])

        return slab, elements

    def render(self, mode="rgb_array"):

        if mode == "rgb_array":
            # return an rgb array representing the picture of the atoms

            # Plot the atoms
            fig, ax1 = plt.subplots()
            plot_atoms(
                self.atoms.repeat((3, 3, 1)),
                ax1,
                rotation="48x,-51y,-144z",
                show_unit_cell=0,
            )

            ax1.set_ylim([0, 25])
            ax1.set_xlim([-2, 20])
            ax1.axis("off")
            ax2 = fig.add_axes([0.35, 0.85, 0.3, 0.1])

            # Add a subplot for the energy history overlay
            ax2.plot(self.history["timesteps"], self.history["energies"])

            ax2.plot(self.minima["timesteps"], self.minima["energies"], "o", color="r")

            if len(self.TS["timesteps"]) > 0:
                ax2.plot(self.TS["timesteps"], self.TS["energies"], "o", color="g")

            ax2.set_ylabel("Energy [eV]")

            # Render the canvas to rgb values for the gym render
            plt.draw()
            renderer = fig.canvas.get_renderer()
            x = renderer.buffer_rgba()
            img_array = np.frombuffer(x, np.uint8).reshape(x.shape)
            plt.close()

            # return the rendered array (but not the alpha channel)
            return img_array[:, :, :3]

        else:
            return

    def close(self):
        return

    def _update_history(self, action_idx, relative_energy):
        self.trajectories.append(self.atoms.copy())
        self.history["timesteps"] = self.history["timesteps"] + [
            self.history["timesteps"][-1] + 1
        ]
        self.history["energies"] = self.history["energies"] + [self.relative_energy]
        self.history["actions"] = self.history["actions"] + [self.action_idx]
        #         self.history['atom_selection'] = self.history['atom_selection'] + [self.atom_selection]
        #         self.history['movement'] = self.history['movement'] + [self.movement]
        self.history["force_calls"] = self.calc.force_calls
        self.history["positions"] = self.history["positions"] + [
            self.atoms.get_positions(wrap=False)[self.free_atoms].tolist()
        ]
        #         self.history['forces'] = self.history['forces'] + [self.forces.tolist()]
        self.history["scaled_positions"] = self.history["scaled_positions"] + [
            self.atoms.get_scaled_positions(wrap=False)[self.free_atoms].tolist()
        ]
        if self.observation_fingerprints:
            self.history["fingerprints"] = self.history["fingerprints"] + [
                self.fps.tolist()
            ]
            self.history["initial_fps"] = self.history["initial_fps"] + [
                self.episode_initial_fps.tolist()
            ]
        return self.history, self.trajectories

    def save_episode(self):
        save_path = os.path.join(
            self.history_dir,
            "%d_%f_%f_%f.npz"
            % (
                self.episodes,
                self.minima["energies"][self.min_idx],
                self.initial_energy,
                self.highest_energy,
            ),
        )
        np.savez_compressed(
            save_path,
            initial_energy=self.initial_energy,
            energies=self.history["energies"],
            actions=self.history["actions"],
            #                  atom_selection = self.history['atom_selection'],
            #                  movement = self.history['movement'],
            #                  positions = self.history['positions'],
            scaled_positions=self.history["scaled_positions"],
            fingerprints=self.history["fingerprints"],
            initial_fps=self.history["initial_fps"],
            #                  scaled_fingerprints = self.history['scaled_fingerprints'],
            minima_energies=self.minima["energies"],
            minima_steps=self.minima["timesteps"],
            minima_TS=self.minima["TS"],
            minima_highest_energyy=self.minima["highest_energy"],
            segregation=self.minima["segregation"],
            #                  TS_energies = self.TS['energies'],
            #                  TS_steps = self.TS['timesteps'],
            force_calls=self.history["force_calls"],
            total_force_calls=self.total_force_calls,
            reward=self.episode_reward,
            atomic_symbols=self.random_free_atoms,
            structure_idx=[self.structure_idx],
            episode=self.episodes,
            #                  forces = self.history['forces'],
        )
        return

    def save_traj(self):

        save_path = os.path.join(
            self.traj_dir,
            "%d_%f_%f_%f_full.traj"
            % (
                self.episodes,
                self.minima["energies"][self.min_idx],
                self.initial_energy,
                self.highest_energy,
            ),
        )
        trajectories = []
        for atoms in self.trajectories:
            atoms.set_calculator(self.ML_calc)
            trajectories.append(atoms)
        write(save_path, trajectories)

        return
