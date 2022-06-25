
import numpy as np


class StarData(object):

    def __init__(self, step, lpi, basis_matrix, input_effects_list=None):
        self.basis_matrix = basis_matrix
        self.input_effects_list = input_effects_list
        self.lpi = lpi
        self.constraints_list = []
        self.n_vars = 2
        self.step = step

    def set_constraints(self, preds):
        self.constraints_list = preds


class LinearPredicate(object):
    'a single linear constraint: vector * x <= value'

    def __init__(self, vector, value):
        self.vector = np.array(vector, dtype=float)
        self.value = float(value)

    def print(self):
        print(str(self.vector), "  <=  " + str(self.value))
