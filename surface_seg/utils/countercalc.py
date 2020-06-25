import os
import copy
import numpy as np

import ase.io
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp

class CounterCalc(Calculator):
    implemented_properties = ["energy", "forces", "uncertainty"]
    """Parameters
    --------------
        calc: object. Parent calculator to track force calls."""

    def __init__(self, calc, **kwargs):
        super().__init__()
        self.calc = calc
        self.force_calls = 0
        
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
#         calc = copy.deepcopy(self.calc)
        calc = self.calc
        self.results["energy"] = calc.get_potential_energy(atoms)
        self.results["forces"] = calc.get_forces(atoms)
        self.force_calls += 1
        
