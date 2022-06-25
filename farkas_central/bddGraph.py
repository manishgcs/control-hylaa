from farkas_central.polytope import Polytope
import numpy as np
from hylaa.timerutil import Timers

# BDD Graph for representing counter examples


class BDDGraphTransition(object):
    def __init__(self, succ_node):
        self.succ_node = succ_node


class OneTransition(BDDGraphTransition):

    def __init__(self, succ_node):
        assert isinstance(succ_node, BDDGraphNode)
        BDDGraphTransition.__init__(self,  succ_node)


class ZeroTransition(BDDGraphTransition):

    def __init__(self, succ_node):
        assert isinstance(succ_node, BDDGraphNode)
        BDDGraphTransition.__init__(self, succ_node)


class BDDGraphNode(object):

    def __init__(self, node_id, level, my_regex='', poly=None):

        self.id = node_id
        self.one_transition = None
        self.zero_transition = None
        self.level = level
        self.my_regex = my_regex
        self.polytope = poly
        self.visited = False

    def new_transition(self, succ_node, t_type):

        if t_type == 0:
            self.zero_transition = ZeroTransition(succ_node)
        elif t_type == 1:
            self.one_transition = OneTransition(succ_node)
        else:
            print("Wrong transition type")


class BDDGraph(object):

    def __init__(self, root_node=None):
        if root_node is None:
            dummy_poly = Polytope(np.asarray([]), np.asarray([]))
            root_node = BDDGraphNode(node_id='r', level=0, my_regex='', poly=dummy_poly)
        self.nodes = [root_node]
        self.root = root_node
        # self.n_layers = 0

    def get_root(self):
        return self.root

    def add_node(self, node):
        self.nodes.append(node)

    def traverse_subtree(self, current_node, current_regex, valid_exps, invalid_exps):
        if current_node.one_transition is None and current_node.zero_transition is None:
            # print(" Terminal node with id " + current_node.id)

            if current_node.id == 't1':
                valid_exps.append(current_regex)
            elif current_node.id == 't0':
                invalid_exps.append(current_regex)
                # print(" \n Expression is " + current_regex)

        else:
            if current_node.one_transition is not None:
                valid_exps, invalid_exps = self.traverse_subtree(current_node.one_transition.succ_node, current_regex+'1',
                                                             valid_exps, invalid_exps)

            if current_node.zero_transition is not None:
                valid_exps, invalid_exps = self.traverse_subtree(current_node.zero_transition.succ_node,
                                                             current_regex + '0', valid_exps, invalid_exps)
        return valid_exps, invalid_exps

    def generate_expressions(self):
        Timers.tic('BDD Traversal Time')
        current_node = self.root
        current_regex = ''
        valid_exps = []
        invalid_exps = []
        valid_exps, invalid_exps = self.traverse_subtree(current_node, current_regex, valid_exps, invalid_exps)
        Timers.toc('BDD Traversal Time')
        return valid_exps, invalid_exps
