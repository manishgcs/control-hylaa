from farkas_central.farkas import FarkasObject
import time
from z3 import *
import numpy as np
from os import environ
from subprocess import Popen
import sys
import subprocess
sys.setrecursionlimit(2000)


def convert_constraints_into_z3_str(c_lhs, c_rhs, alpha_or_y='alpha', r_operator='<='):
    z3_constraints = ''

    for idx in range(len(c_lhs)):
        if z3_constraints is not '':
            z3_constraints = z3_constraints + ', '
        z3_pred_str = ''
        row = c_lhs[idx]
        for idz in range(len(row)):
            if row[idz] != 0:
                if z3_pred_str is not '':
                    z3_pred_str = z3_pred_str + ' + '

                z3_pred_str = z3_pred_str + str(row[idz]) + '*' + alpha_or_y + '_' + str(idz + 1)

        z3_pred_str = z3_pred_str + ' ' + r_operator + ' ' + str(c_rhs[idx])
        z3_constraints = z3_constraints + z3_pred_str
    return z3_constraints


def check_poly_feasibility_smt(poly):
    s = Solver()
    set_option(rational_to_decimal=True)

    alpha = []
    for dim in range(poly.n_state_vars):
        alpha_i = 'alpha_' + str(dim + 1)
        alpha.append(Real(alpha_i))

    c_idx = True
    for idy in range(poly.n_constraints):
        c_idy = alpha[0] * poly.con_matrix[idy][0]
        for idz in range(1, poly.n_state_vars):
            c_idy = c_idy + alpha[idz] * poly.con_matrix[idy][idz]

        c_idx = And(c_idx, c_idy <= poly.rhs[idy])

    s.add(c_idx)

    alpha_vals = []
    if s.check() == sat:
        mdl = s.model()

        for idx in range(poly.n_state_vars):
            alpha_vals.append(mdl[alpha[idx]])

    return alpha_vals


class FarkasSMT(FarkasObject):

    def __init__(self, P1, P2, Q_set, n_vars):
        FarkasObject.__init__(self, P1, P2, Q_set, n_vars)

    def solve_4_both_smt_file(self):
        start_time = time.time()
        file_name = "farkas_z3_preds.py"

        n_z_vars = 1 + len(self.q_set)

        A1_matrix = self.polytope1.con_matrix
        A2_matrix = self.polytope2.con_matrix
        b1_array = self.polytope1.rhs
        b2_array = self.polytope2.rhs

        n1_constraints_per_polytope = [len(self.polytope1.rhs)]
        n2_constraints_per_polytope = [len(self.polytope2.rhs)]

        for idx in range(len(self.q_set)):
            q_poly = self.q_set[idx]
            A1_matrix = np.concatenate((A1_matrix, q_poly.con_matrix))
            A2_matrix = np.concatenate((A2_matrix, q_poly.con_matrix))
            b1_array = np.concatenate((b1_array, q_poly.rhs))
            b2_array = np.concatenate((b2_array, q_poly.rhs))
            n1_constraints_per_polytope.append(len(q_poly.rhs))
            n2_constraints_per_polytope.append(len(q_poly.rhs))

        assert n_z_vars == len(n1_constraints_per_polytope)
        assert len(A1_matrix) == np.sum(n1_constraints_per_polytope)

        with open(file_name, 'w') as z3_preds_file:
            z3_preds_file.write('from z3 import *\n\n')
            for dim in range(self.n_state_vars):
                alpha_i = 'alpha_' + str(dim + 1)
                z3_preds_file.write('{} = Real(\'{}\')\n'.format(alpha_i, alpha_i))

            n_y_vars = 0
            # y >= 0
            for idx in range(len(n2_constraints_per_polytope)):
                n2_constraints = n2_constraints_per_polytope[idx]
                for idy in range(n2_constraints):
                    y_i = 'y_' + str(n_y_vars + 1)
                    z3_preds_file.write('{} = Real(\'{}\')\n'.format(y_i, y_i))
                    n_y_vars = n_y_vars + 1

            assert n_y_vars == len(A2_matrix)

            z3_preds_file.write('s = Optimize()\n')
            z3_preds_file.write('set_option(rational_to_decimal=True)\n')

            ida = 0
            constraint_id = 1
            for idx in range(n_z_vars):
                n1_constraints = n1_constraints_per_polytope[idx]
                # print(ida, n1_constraints)
                # print(A1_matrix[ida:n1_constraints])
                c_lhs = A1_matrix[ida:ida+n1_constraints]
                c_rhs = b1_array[ida:ida+n1_constraints]
                # print(ida, c_lhs, c_rhs)
                z3_constraints = convert_constraints_into_z3_str(c_lhs, c_rhs)
                c_i = 'c_' + str(constraint_id)
                z3_preds_file.write('{} = Bool(\'{}\')\n'.format(c_i, c_i))
                z3_preds_file.write('s.add({} == And({}))\n'.format(c_i, z3_constraints))
                z3_preds_file.write('s.add_soft({})\n'.format(c_i))
                ida = ida + n1_constraints
                constraint_id = constraint_id + 1
            z3_preds_file.write('s.add(c_1 == True)\n')

            n_y_vars = 0
            # y >= 0
            for idx in range(len(n2_constraints_per_polytope)):
                n2_constraints = n2_constraints_per_polytope[idx]
                for idy in range(n2_constraints):
                    y_i = 'y_' + str(n_y_vars + 1)
                    z3_preds_file.write('s.add({} >= 0.0)\n'.format(y_i))
                    n_y_vars = n_y_vars + 1

            A2_matrix_t = np.transpose(A2_matrix)

            # A^T * y = 0
            for idx in range(len(A2_matrix_t)):
                A2_t_row = A2_matrix_t[idx]
                z3_constraints = convert_constraints_into_z3_str([A2_t_row], [0], 'y', ' ==')
                z3_preds_file.write('s.add({})\n'.format(z3_constraints))

            # b^T * y < 0
            b2_array_t = np.transpose(b2_array)
            assert n_y_vars == len(b2_array_t)
            z3_constraints = convert_constraints_into_z3_str([b2_array_t], [0], 'y', '<')
            z3_preds_file.write('s.add({})\n'.format(z3_constraints))

            # y = 0 for z = 0, y > 0 for z = 1
            y_var_idx = n2_constraints_per_polytope[0]  # First entry corresponds to P2. The rest of them are for Q-set
            for idx in range(1, len(n2_constraints_per_polytope)):
                n2_constraints = n2_constraints_per_polytope[idx]
                for idy in range(n2_constraints):
                    z3_pred_str = 'Or(And(c_' + str(idx+1) + ' == False, ' + 'y_' + str(y_var_idx+1) + '== 0.0), '
                    z3_pred_str = z3_pred_str + 'And(c_' + str(idx+1) + ' == True, ' + 'y_' + str(y_var_idx+1) + ' > 0.0))'
                    # z3_pred_str = 'Or(c_' + str(idx+1) + ' == True, ' + 'y_' + str(y_var_idx+1) + '== 0.0)'
                    z3_preds_file.write('s.add({})\n'.format(z3_pred_str))
                    y_var_idx = y_var_idx + 1

            z3_preds_file.write('if s.check() == sat:\n')
            z3_preds_file.write('\tm = s.model()\n')
            z3_preds_file.write('\tprint(m)\n')
            z3_preds_file.write('else:\n')
            z3_preds_file.write('\tprint(\u0027No solution\u0027)')
        z3_preds_file.close()
        env = dict(environ)
        args = ['python3', file_name]
        p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        for line in p.stdout.readlines():
            print(line)
        z3_preds_file.close()
        print("Time taken by SMT: {}".format(str(time.time() - start_time)))

    def solve_4_both(self):
        start_time = time.time()
        n_z_vars = 1 + len(self.q_set)

        A1_matrix = self.polytope1.con_matrix
        A2_matrix = self.polytope2.con_matrix
        b1_array = self.polytope1.rhs
        b2_array = self.polytope2.rhs

        n1_constraints_per_polytope = [len(self.polytope1.rhs)]
        n2_constraints_per_polytope = [len(self.polytope2.rhs)]

        for idx in range(len(self.q_set)):
            q_poly = self.q_set[idx]
            A1_matrix = np.concatenate((A1_matrix, q_poly.con_matrix))
            A2_matrix = np.concatenate((A2_matrix, q_poly.con_matrix))
            b1_array = np.concatenate((b1_array, q_poly.rhs))
            b2_array = np.concatenate((b2_array, q_poly.rhs))
            n1_constraints_per_polytope.append(len(q_poly.rhs))
            n2_constraints_per_polytope.append(len(q_poly.rhs))

        assert n_z_vars == len(n1_constraints_per_polytope)
        assert len(A1_matrix) == np.sum(n1_constraints_per_polytope)

        s = Optimize()
        set_option(rational_to_decimal=True)

        alpha = []
        for dim in range(self.n_state_vars):
            alpha_i = 'alpha_' + str(dim + 1)
            alpha.append(Real(alpha_i))

        n_y_vars = 0
        # y >= 0
        y = []
        for idx in range(len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            for idy in range(n2_constraints):
                y_i = 'y_' + str(n_y_vars + 1)
                y.append(Real(y_i))
                n_y_vars = n_y_vars + 1

        assert n_y_vars == len(A2_matrix)

        # constraints for decision variables z's
        ida = 0
        z = []
        for idx in range(n_z_vars):
            n1_constraints = n1_constraints_per_polytope[idx]
            c_lhs = A1_matrix[ida:ida + n1_constraints]
            c_rhs = b1_array[ida:ida + n1_constraints]

            z_idx = 'z_' + str(idx + 1)
            z.append(Bool(z_idx))

            c_idx = True
            for idy in range(n1_constraints):
                c_idy = alpha[0] * c_lhs[idy][0]

                for idz in range(1, self.n_state_vars):
                    c_idy = c_idy + alpha[idz] * c_lhs[idy][idz]

                c_idx = And(c_idx, c_idy <= c_rhs[idy])

            s.add(z[idx] == c_idx)
            s.add_soft(z[idx])
            ida = ida + n1_constraints

        s.add(z[0] == True)

        # y >= 0
        n_y_vars = 0
        for idx in range(len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            for idy in range(n2_constraints):
                s.add(y[n_y_vars] >= 0.0)
                n_y_vars = n_y_vars + 1

        A2_matrix_t = np.transpose(A2_matrix)

        # A^T * y = 0
        for idx in range(len(A2_matrix_t)):
            A2_t_row = A2_matrix_t[idx]

            c_idx = A2_t_row[0] * y[0]

            for idy in range(1, len(A2_t_row)):
                c_idx = c_idx + A2_t_row[idy] * y[idy]

            s.add(c_idx == 0.0)

        # b^T * y < 0
        b2_array_t = np.transpose(b2_array)
        assert n_y_vars == len(b2_array_t)

        c_idx = b2_array_t[0] * y[0]

        for idy in range(1, len(b2_array_t)):
            c_idx = c_idx + b2_array_t[idy] * y[idy]

        s.add(c_idx < 0.0)

        # y = 0 for z = 0, y > 0 for z = 1
        y_var_idx = n2_constraints_per_polytope[0]  # First entry corresponds to P2. The rest of them are for Q-set
        for idx in range(1, len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            for idy in range(n2_constraints):
                s.add(Or(And(z[idx] == False, y[y_var_idx] == 0.0), And(z[idx] == True, y[y_var_idx] > 0.0)))
                # s.add(Or(z[idx] == True, y[y_var_idx] == 0.0))
                y_var_idx = y_var_idx + 1

        z_vals = []
        alpha_vals = []

        if s.check() == sat:
            mdl = s.model()

            for idx in range(n_z_vars):
                z_vals.append(mdl[z[idx]])

            for idx in range(self.n_state_vars):
                alpha_vals.append(mdl[alpha[idx]])

        print("Time taken by SMT: {}".format(str(time.time() - start_time)))

        return z_vals, alpha_vals

    # Making z2 (second polytope explicitly true) - Doesn't seem to be working. In fact, it gives some weird results
    def solve_4_both_test(self):
        start_time = time.time()
        n_z_vars = 1 + len(self.q_set)

        A1_matrix = self.polytope1.con_matrix
        A2_matrix = None
        b1_array = self.polytope1.rhs
        b2_array = None

        n1_constraints_per_polytope = [len(self.polytope1.rhs)]
        n2_constraints_per_polytope = []

        for idx in range(len(self.q_set)):
            q_poly = self.q_set[idx]
            A1_matrix = np.concatenate((A1_matrix, q_poly.con_matrix))
            if A2_matrix is None:
                A2_matrix = q_poly.con_matrix
            else:
                A2_matrix = np.concatenate((A2_matrix, q_poly.con_matrix))
            b1_array = np.concatenate((b1_array, q_poly.rhs))

            if b2_array is None:
                b2_array = q_poly.rhs
            else:
                b2_array = np.concatenate((b2_array, q_poly.rhs))
            n1_constraints_per_polytope.append(len(q_poly.rhs))
            n2_constraints_per_polytope.append(len(q_poly.rhs))

        assert n_z_vars == len(n1_constraints_per_polytope)
        assert len(A1_matrix) == np.sum(n1_constraints_per_polytope)

        s = Optimize()
        set_option(rational_to_decimal=True)

        alpha = []
        for dim in range(self.n_state_vars):
            alpha_i = 'alpha_' + str(dim + 1)
            alpha.append(Real(alpha_i))

        n_y_vars = 0
        # y >= 0
        y = []
        for idx in range(len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            for idy in range(n2_constraints):
                y_i = 'y_' + str(n_y_vars + 1)
                y.append(Real(y_i))
                n_y_vars = n_y_vars + 1

        assert n_y_vars == len(A2_matrix)

        # constraints for decision variables z's
        ida = 0
        z = []
        for idx in range(n_z_vars):
            n1_constraints = n1_constraints_per_polytope[idx]
            c_lhs = A1_matrix[ida:ida + n1_constraints]
            c_rhs = b1_array[ida:ida + n1_constraints]

            z_idx = 'z_' + str(idx + 1)
            z.append(Bool(z_idx))

            c_idx = True
            for idy in range(n1_constraints):
                c_idy = alpha[0] * c_lhs[idy][0]

                for idz in range(1, self.n_state_vars):
                    c_idy = c_idy + alpha[idz] * c_lhs[idy][idz]

                c_idx = And(c_idx, c_idy <= c_rhs[idy])

            s.add(z[idx] == c_idx)
            s.add_soft(z[idx])
            ida = ida + n1_constraints

        s.add(z[0] == True)

        # constraints for decision variables z's
        c_lhs = self.polytope2.con_matrix
        c_rhs = self.polytope2.rhs

        z2 = Bool('z2')

        c_idx = True
        for idy in range(len(self.polytope2.rhs)):
            c_idy = alpha[0] * c_lhs[idy][0]

            for idz in range(1, self.n_state_vars):
                c_idy = c_idy + alpha[idz] * c_lhs[idy][idz]

            c_idx = And(c_idx, c_idy <= c_rhs[idy])

        s.add(True == c_idx)

        s.add(z[0] == True)

        # y >= 0
        n_y_vars = 0
        for idx in range(len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            for idy in range(n2_constraints):
                s.add(y[n_y_vars] >= 0.0)
                n_y_vars = n_y_vars + 1

        A2_matrix_t = np.transpose(A2_matrix)

        # A^T * y = 0
        for idx in range(len(A2_matrix_t)):
            A2_t_row = A2_matrix_t[idx]

            c_idx = A2_t_row[0] * y[0]

            for idy in range(1, len(A2_t_row)):
                c_idx = c_idx + A2_t_row[idy] * y[idy]

            s.add(c_idx == 0.0)

        # b^T * y < 0
        b2_array_t = np.transpose(b2_array)
        assert n_y_vars == len(b2_array_t)

        c_idx = b2_array_t[0] * y[0]

        for idy in range(1, len(b2_array_t)):
            c_idx = c_idx + b2_array_t[idy] * y[idy]

        s.add(c_idx < 0.0)

        # y = 0 for z = 0, y > 0 for z = 1
        y_var_idx = 0
        # y_var_idx = n2_constraints_per_polytope[0]  # First entry corresponds to P2. The rest of them are for Q-set
        for idx in range(0, len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            for idy in range(n2_constraints):
                s.add(Or(And(z[idx] == False, y[y_var_idx] == 0.0), And(z[idx] == True, y[y_var_idx] > 0.0)))
                # s.add(Or(z[idx] == True, y[y_var_idx] == 0.0))
                y_var_idx = y_var_idx + 1

        z_vals = []
        alpha_vals = []

        if s.check() == sat:
            mdl = s.model()

            for idx in range(n_z_vars):
                z_vals.append(mdl[z[idx]])

            for idx in range(self.n_state_vars):
                alpha_vals.append(mdl[alpha[idx]])

        print("Time taken by SMT: {}".format(str(time.time() - start_time)))

        return z_vals, alpha_vals

