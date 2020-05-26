import gym
from gym import error, spaces, utils
from gym.utils import seeding


MOVE_ACTION_NAMES = [
    'up', 
    'down', 
    'left', 
    'right', 
    'forward', 
    'backward']

MOVE_ACTION = [
    np.array([ 0, 0, 1]), 
    np.array([ 0, 0,-1]), 
    np.array([ 1, 0, 0]), 
    np.array([-1, 0, 0]), 
    np.array([ 0, 1, 0]), 
    np.array([ 0,-1, 0])]

ACTION_LOOKUP = [
    'move',
    'minimize',
    'transition_state_search',
    'steepest_descent', 
    'steepest_ascent']

class MultiComponentSurface(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.initial_atoms = self._generate_slab()

        #Mark the free atoms
        self.free_atoms = list(set(range(len(self.initial_atoms))) - 
            set(self.initial_atoms.constraints[0].get_indices()))
        
        #Define the possible actions        
        self.action_space = spaces.Tuple((spaces.Discrete(len(ACTION_TYPE)),
                                          spaces.Discrete(len(self.free_atoms)),
                                          spaces.Discrete(len(MOVE_ACTION))))
        
        #Set up the initial atoms
        self.reset()
        
        return

    #open AI gym API requirement
    def step(self, action):
    
        action_type = ACTION_LOOKUP[action[0]]
        
        if action_type=='move':
            atom_index = ACTION_LOOKUP[action[1]]
            move_index = ACTION_LOOKUP[action[2]]
            self._move_atom(atom_index, move_index)
            
        elif self.action_type=='minimize':
            self._minimize()

        elif self.action_type=='transition_state_search':
            self._transition_state_search()
            
        elif self.action_type=='steepest_descent':
            atom_index = ACTION_LOOKUP[action[1]]
            self._steepest_descent(atom_index)
            
        elif self.action_type=='steepest_ascent':
            atom_index = ACTION_LOOKUP[action[1]]
            self._steepest_ascent(atom_index)
            
        else:
            raise Exception('I am not sure what action you mean!')
            
        observation = self._get_state()
        reward = self._get_reward()
        episode_over=False
        
        return self._get_state()

    #open AI gym API requirement
    def reset(self):
        self.atoms = self.initial_atoms.copy()
        return self._get_state()
    
    #open AI gym API requirement
    def render(self, mode='human'):
        # Fill in with ASE visualization
        return
    
    #open AI gym API requirement
    def close(self):
        return
    
        
    
    #Helper functions for moves
    def _minimize(self):
        dyn = BFGSLineSearch(atoms=self.atoms, logfile=None)
        dyn.run(0.1)
        return
    
    def _get_reward(self):
        # update this
        return atoms.get_potential_energy()
    
    def _transition_state_search(self):
        fix = self.atoms.constraints[0].get_indices()

        dyn = Sella(self.atoms,  # Your Atoms object
                         constraints=dict(fix=fix),  # Your constraints
                         trajectory='saddle.traj',  # Optional trajectory
                         )
        dyn.run(1e-2, steps = 100)
            
        return 
    
    def _steepest_descent(self, atom_idx):
        force = self.atoms.get_forces()[self.free_atoms,:]
        move = -0.1*force[atom_idx]
        return 
    
    def _steepest_ascent(self):
        force = self.atoms.get_forces()[self.free_atoms,:]
        move = 0.1*force[atom_idx]
        return 
    
    def _move_atom(self, atom_index, move_index):
        #Helper function to move an atom
        self.atoms.positions[self.free_atoms[atom_index]]+=MOVE_ACTION[move_index]
        return
    
    def _get_state(self):
        # helper function to get the current state space, which is just the position
        # of the free atoms as one long vector (should be improved)
        return self.positions()[self.free_atoms].reshape((-1,1))
    
    def _generate_slab(self):
        #generate a pseudo-random sequence of elements
        np.random.seed(permute_seed)
        num_atoms = math.prod(size)
        atom_ordering = list(itertools.chain.from_iterable([[key]*element_choices[key] for key in element_choices]))
        element_list = np.random.permutation(atom_ordering)

        #Use vergard's law to estimate the lattice constant
        a = np.sum([element_lattice_constants[key]*element_choices[key]/num_atoms for key in element_choices])

        #Generate a base FCC slab
        slab = fcc111('Al', size=size, a=a, periodic=True, vacuum=10.0)
        slab.set_chemical_symbols(element_list)

        #Set the calculator
        slab.set_calculator(EMT())

        #Constrain the bottom two layers
        c =  FixAtoms(indices=[atom.index for atom in slab if atom.position[2] < np.mean(slab.positions[:,2])])  #Fix two layers
        slab.set_constraint(c)
        
        #Do a quick minimization to relax the structure
        dyn = BFGSLineSearch(atoms=slab, logfile=None)
        dyn.run(0.1)
        
        return slab