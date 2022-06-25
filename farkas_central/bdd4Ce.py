from farkas_central.farkas_mip import FarkasMIP, check_poly_feasibility
from farkas_central.farkas_smt import FarkasSMT, check_poly_feasibility_smt
import numpy as np
from farkas_central.polytope import Polytope
from farkas_central.star_data import LinearPredicate
from farkas_central.bddGraph import BDDGraphNode, BDDGraph
from hylaa.timerutil import Timers
from memory_profiler import profile
import random


def compute_usafe_set_pred_in_star_basis(error_star, usafeset_preds):
    usafe_basis_predicates = []

    # print(error_star.constraint_list)
    for std_pred in usafeset_preds:
        # Translating usafe set star into the simulation/star's basis
        # std_pred.print()
        # print(error_star.basis_matrix)
        new_lc_vector = np.dot(std_pred.vector, error_star.basis_matrix)
        # new_lc_value = std_pred.value - np.dot(error_star.center, std_pred.vector)
        new_lc_value = std_pred.value
        new_pred = LinearPredicate(new_lc_vector, new_lc_value)
        # new_pred.print()
        usafe_basis_predicates.append(new_pred)

    for constraint in error_star.constraints_list:
        usafe_basis_predicates.append(constraint)
        # constraint.print()

    return usafe_basis_predicates


class BDD4CE(object):
    def __init__(self, error_states, usafe_set_preds, equ_run=True, smt_mip='mip'):
        self.error_states = error_states
        self.usafe_set = usafe_set_preds
        self.n_state_vars = 2
        self.n_paths = 1  # don't need it unless we extend it to hybrid systems
        self.p_intersect_u = []
        self.p_intersect_not_u = []
        self.equ_run = equ_run
        self.order_idx = []
        self.smt_mip = smt_mip

    # def clear_lists(self):
    #     self.n_paths = 1
    #     self.p_intersect_u = []
    #     self.p_intersect_not_u = []

    def partitiion_error_stars_wrt_usafe_boundary(self):

        # self.clear_lists()
        self.n_state_vars = self.error_states[0].basis_matrix.shape[1]

        p_intersect_u_in_path = []
        p_intersect_not_u_in_path = []
        for error_state in self.error_states:

            usafe_basis_preds = compute_usafe_set_pred_in_star_basis(error_state, self.usafe_set)

            no_of_constraints = len(usafe_basis_preds)
            con_matrix = np.zeros((no_of_constraints, self.n_state_vars), dtype=float)
            rhs = np.zeros(no_of_constraints, dtype=float)
            for idx in range(len(usafe_basis_preds)):
                pred = usafe_basis_preds[idx]
                pred_list = pred.vector.tolist()
                # print(pred_list)
                for idy in range(len(pred_list)):
                    if pred_list[idy] != 0:
                        con_matrix[idx][idy] = pred_list[idy]
                rhs[idx] = pred.value
                # print(con_matrix)
            polytope_u = Polytope(con_matrix, rhs)
            p_intersect_u_in_path.append(polytope_u)

            safe_constraint_list = []
            # print(self.usafe_set_constraint_list)
            for pred in self.usafe_set:
                lc_vector = -1 * pred.vector
                lc_value = -1 * (pred.value + 0.000001)
                safe_constraint_list.append(LinearPredicate(lc_vector, lc_value))

            safe_basis_preds = compute_usafe_set_pred_in_star_basis(error_state, safe_constraint_list)

            no_of_constraints = len(safe_basis_preds)
            con_matrix = np.zeros((no_of_constraints, self.n_state_vars), dtype=float)
            rhs = np.zeros(no_of_constraints, dtype=float)
            for idx in range(len(safe_basis_preds)):
                pred = safe_basis_preds[idx]
                pred_list = pred.vector.tolist()
                for idy in range(len(pred_list)):
                    if pred_list[idy] != 0:
                        con_matrix[idx][idy] = pred_list[idy]
                rhs[idx] = pred.value
            # print(con_matrix)
            polytope_not_u = Polytope(con_matrix, rhs)
            p_intersect_not_u_in_path.append(polytope_not_u)

        # print(len(p_intersect_u_in_path), len(self.p_intersect_u))
        self.p_intersect_u.append(p_intersect_u_in_path)
        self.p_intersect_not_u.append(p_intersect_not_u_in_path)

        assert len(self.p_intersect_u) == self.n_paths
        assert len(self.p_intersect_not_u) == self.n_paths

    # test_instance_smt = FarkasSMT(p1, p2, q_set, self.n_state_vars)
    # z_vals, alpha_vals = test_instance_smt.solve_4_both_smt()
    # print(z_vals, alpha_vals)
    # test_instance_smt = FarkasSMT(p2, p1, q_set, self.n_state_vars)
    # z_vals, alpha_vals = test_instance_smt.solve_4_both_smt()
    # print(z_vals, alpha_vals)

    # test_instance_mip.solve_4_one_polytope_mip(poly_idx=2)
    # z_vals, alpha_vals = test_instance_mip.solve_4_both_mip()
    # print(z_vals, alpha_vals)
    # test_instance_mip = FarkasMIP(p2, p1, q_set, self.n_state_vars)
    # z_vals, alpha_vals = test_instance_mip.solve_4_both_mip()
    # print(z_vals, alpha_vals)

    def check_equivalence_among_set(self, poly1, node_set, q1_set, q2_set):
        equivalent_node = None

        if len(node_set) == 0:
            return equivalent_node

        print("no of nodes in the set " + str(len(node_set)))
        for node2 in node_set:
            print(node2.id)
            z_vals = self.check_equivalence(poly1, node2.polytope, q1_set, q2_set)
            print(z_vals)
            if not z_vals:
                equivalent_node = node2
                break

        return equivalent_node

    def check_equivalence(self, poly1, poly2, q1_set, q2_set):

        # A non-empty return value means two polytopes are not equivalent
        if self.equ_run is False:
            return [1]

        if self.smt_mip == 'mip':
            test_instance_1 = FarkasMIP(poly1, poly2, q1_set, self.n_state_vars)
            test_instance_2 = FarkasMIP(poly2, poly1, q1_set, self.n_state_vars)
            test_instance_3 = FarkasMIP(poly2, poly1, q2_set, self.n_state_vars)
            test_instance_4 = FarkasMIP(poly1, poly2, q2_set, self.n_state_vars)
        else:
            test_instance_1 = FarkasSMT(poly1, poly2, q1_set, self.n_state_vars)
            test_instance_2 = FarkasSMT(poly2, poly1, q1_set, self.n_state_vars)
            test_instance_3 = FarkasSMT(poly2, poly1, q2_set, self.n_state_vars)
            test_instance_4 = FarkasSMT(poly1, poly2, q2_set, self.n_state_vars)

        z_vals, alpha_vals = test_instance_1.solve_4_both()

        print(" *** Tested instance 1 **** ")

        if not z_vals:
            z_vals, alpha_vals = test_instance_2.solve_4_both()

            print(" *** Tested instance 2 **** ")

            if not z_vals:
                z_vals, alpha_vals = test_instance_3.solve_4_both()

                print(" *** Tested instance 3 **** ")

                if not z_vals:
                    z_vals, alpha_vals = test_instance_4.solve_4_both()

                    print(" *** Tested instance 4 **** ")

        return z_vals

    # Equivalence is checked between p and not p at each step. Additionally, the bdd is tested for equivalence
    # among all nodes at a user provided level.
    # level_merge = -1 means not to carry out level wise merging
    # level_merge = 0 denotes level merge at each level
    # level_merge = r > 0 depicts merging of nodes only at level r
    # @profile(precision=4)
    def create_bdd_w_level_merge(self, level_merge=-1, order='default'):
        nodes_count = []
        unique_states = []
        Timers.tic('BDD Construction New')

        self.partitiion_error_stars_wrt_usafe_boundary()
        bdd_graphs = []

        for path_idx in range(self.n_paths):

            p_intersect_u_in_path = self.p_intersect_u[path_idx]
            p_intersect_not_u_in_path = self.p_intersect_not_u[path_idx]

            n_polytopes = len(p_intersect_u_in_path)
            if level_merge >= n_polytopes:
                print("\n ******** Please specify a level between 1 and {} for merging *********** \n ".format(str(n_polytopes-1)))
                continue

            order_idx = list(range(n_polytopes))
            if order == 'reverse':
                order_idx = order_idx[::-1]
            elif order == 'random':
                random.Random(20).shuffle(order_idx)
            elif order == 'mid-order':
                if n_polytopes <= 10:
                    mid_element = 1
                elif n_polytopes <= 20:
                    mid_element = 2
                else:
                    mid_element = 3
                second_half = order_idx[int(n_polytopes / 2)+mid_element:]
                first_half = order_idx[:int(n_polytopes / 2)+mid_element]
                # second_half = second_half[::-1]
                first_half = first_half[::-1]
                first_half.extend(second_half)
                order_idx = first_half
                # random.Random(20).shuffle(order_idx)
                print(first_half, second_half)
                print(order_idx)

                # r_idx = random.sample(r_idx, len(r_idx))

            self.order_idx = order_idx

            # r_idx = [2, 0, 1, 3, 4]  # Works for [0, 2, 1, 3, 4] as well
            # order_idx = [2, 0, 1, 3, 4]

            if order != 'default':
                print(order_idx)
                p_intersect_u = []
                p_intersect_not_u = []
                for idx in order_idx:
                    p_intersect_u.append(p_intersect_u_in_path[idx])
                    p_intersect_not_u.append(p_intersect_not_u_in_path[idx])

                p_intersect_u_in_path = p_intersect_u
                p_intersect_not_u_in_path = p_intersect_not_u

            queue_bdd_nodes = []

            bdd_graph = BDDGraph()
            bdd_root = bdd_graph.get_root()

            bdd_node_t0 = BDDGraphNode(node_id='t0', level=n_polytopes, my_regex='t0')
            bdd_node_t1 = BDDGraphNode(node_id='t1', level=n_polytopes, my_regex='t1')

            queue_bdd_nodes.append(bdd_root)
            nodes_count.append(1)

            current_level = 0

            while current_level < n_polytopes:

                n_nodes_at_next_level = 1  # to incrementally assign id's for labeling the nodes at current level
                nodes_at_next_level = []

                p1 = p_intersect_u_in_path[current_level]
                not_p1 = p_intersect_not_u_in_path[current_level]
                q1_set = p_intersect_u_in_path[(current_level+1):n_polytopes]
                q2_set = p_intersect_not_u_in_path[(current_level+1):n_polytopes]

                while queue_bdd_nodes:
                    current_node = queue_bdd_nodes.pop(0)
                    current_node_regex = current_node.my_regex  # regex is just for debugging and printing the path
                    current_p = current_node.polytope

                    p_intersect_p1 = current_p.intersect_w_q(p1)
                    p_intersect_not_p1 = current_p.intersect_w_q(not_p1)

                    # To check, whether either or both polytopes are infeasible

                    if self.smt_mip == 'smt':
                        alpha_vals1 = check_poly_feasibility_smt(p_intersect_p1)
                        alpha_vals2 = check_poly_feasibility_smt(p_intersect_not_p1)
                    else:
                        alpha_vals1 = check_poly_feasibility(p_intersect_p1)
                        alpha_vals2 = check_poly_feasibility(p_intersect_not_p1)

                    if not alpha_vals1:  # If first is infeasible

                        print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '1')))
                        current_node.new_transition(bdd_node_t0, 1)

                    if not alpha_vals2:  # If second is infeasible
                        print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '0')))
                        current_node.new_transition(bdd_node_t0, 0)

                    if alpha_vals1:  # If first is feasible
                        print(" \n **** {} is feasible **** \n".format((current_node_regex + '1')))

                        if current_level != n_polytopes - 1:  # Add these nodes/polytopes only if not the last level

                            equ_node = None
                            if level_merge == 0 or (current_level+1 == level_merge):
                                equ_node = self.check_equivalence_among_set(p_intersect_p1, nodes_at_next_level,
                                                                            q1_set, q2_set)

                            if equ_node is None:
                                print(" equivalent node not found ")
                                node_id = str(current_level + 1) + str(n_nodes_at_next_level)
                                n_nodes_at_next_level = n_nodes_at_next_level + 1
                                bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                        my_regex=current_node_regex + '1', poly=p_intersect_p1)
                                current_node.new_transition(bdd_node, 1)
                                # queue_bdd_nodes.append(bdd_node)

                                # if level_merge != -1:
                                nodes_at_next_level.append(bdd_node)
                            else:
                                print("equivalent node is found")
                                current_node.new_transition(equ_node, 1)
                        else:  # terminal level
                            unique_states.append(alpha_vals1)
                            current_node.new_transition(bdd_node_t1, 1)  # Make transition on label 1 to terminal 1

                    if alpha_vals2:  # If second is feasible
                        print(" \n **** {} is feasible **** \n".format((current_node_regex + '0')))

                        if current_level != n_polytopes - 1:  # Add these nodes/polytopes only if not the last level

                            equ_node = None
                            if level_merge == 0 or (current_level+1 == level_merge):
                                equ_node = self.check_equivalence_among_set(p_intersect_not_p1, nodes_at_next_level,
                                                                                q1_set, q2_set)

                            if equ_node is None:
                                print("equivalent node not found")
                                node_id = str(current_level + 1) + str(n_nodes_at_next_level)
                                n_nodes_at_next_level = n_nodes_at_next_level + 1
                                bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                        my_regex=current_node_regex + '0', poly=p_intersect_not_p1)
                                current_node.new_transition(bdd_node, 0)

                                # queue_bdd_nodes.append(bdd_node)

                                # if level_merge != -1:
                                nodes_at_next_level.append(bdd_node)
                            else:
                                print("equivalent node is found")
                                current_node.new_transition(equ_node, 0)
                        else:  # terminal level
                            unique_states.append(alpha_vals2)
                            current_node.new_transition(bdd_node_t1, 0)  # Make transition on label 0 to terminal 1

                if nodes_at_next_level:
                    nodes_count.append(len(nodes_at_next_level))
                current_level = current_level + 1
                print(" current level is " + str(current_level))
                queue_bdd_nodes = nodes_at_next_level

            bdd_graphs.append(bdd_graph)
        nodes_count.append(2)  # terminal nodes

        if nodes_count:
            print(nodes_count, sum(nodes_count))
            print(self.order_idx)
            print(len(unique_states))
            # print(unique_states)

            # print(r_idx)
        # print(unique_states)
        Timers.toc('BDD Construction New')
        return bdd_graphs
