from farkas_central.farkas import FarkasObject
import time
import numpy as np
from gurobipy import *


def check_poly_feasibility(poly):
    model = Model("CheckFeasibility")

    # Oscillator
    lb = -10000.0
    ub = 1000.0

    alpha = []
    for idx in range(poly.n_state_vars):
        temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="alpha[" + str(idx) + "]")
        alpha.append(temp_var)
    model.update()

    for idy in range(poly.n_constraints):
        con = LinExpr(0.0)
        for idz in range(poly.n_state_vars):
            con += poly.con_matrix[idy][idz] * alpha[idz]
        model.addConstr(con <= poly.rhs[idy], name="pcon[" + str(idy) + "]")

    objective = LinExpr(0.0)

    model.setObjective(objective, GRB.MAXIMIZE)

    model.setParam(GRB.Param.Threads, 8)
    model.setParam(GRB.Param.TimeLimit, 100.0)
    model.setParam(GRB.Param.MIPGap, 1e-7)
    # model.setParam(GRB.Param.OptimalityTol, 1e-9)
    # model.setParam(GRB.Param.FeasibilityTol, 1e-9)
    # model.setParam(GRB.Param.IntFeasTol, 1e-9)

    model.optimize()

    status = model.Status
    alpha_vals = []
    if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
            status == GRB.UNBOUNDED):
        print("Optimization stopped with status " + str(status))

    else:
        print("Bounds:\t" + str(int(model.objVal)) + "\t" + str(int(model.objBound)))

        for idx in range(poly.n_state_vars):
            alpha_vals.append(float(alpha[idx].getAttr(GRB.Attr.X)))

    return alpha_vals


class FarkasMIP(FarkasObject):

    def __init__(self, P1, P2, Q_set, n_vars):
        FarkasObject.__init__(self, P1, P2, Q_set, n_vars)

    def solve_4_one_polytope_mip(self, poly_idx):
        model = Model("TestOnePolytope")

        n_z_vars = 1 + len(self.q_set)

        # Oscillator
        bigM = 10000.0
        lb = -1000.0
        ub = 100.0

        # Ball
        # bigM = 1000.0
        # lb = -1000.0
        # ub = 10.0

        x = []
        for idx in range(self.n_state_vars):
            temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="x[" + str(idx) + "]")
            x.append(temp_var)
        model.update()

        z = []
        for idx in range(n_z_vars):
            temp_var = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="z[" + str(idx) + "]")
            z.append(temp_var)
        model.update()

        for idx in range(n_z_vars):
            if idx == 0 and poly_idx == 1:
                polytope = self.polytope1
            elif idx == 0 and poly_idx == 2:
                polytope = self.polytope2
            else:
                polytope = self.q_set[idx-1]
            for idy in range(polytope.n_constraints):
                con = LinExpr(0.0)
                for idz in range(self.n_state_vars):
                    con += polytope.con_matrix[idy][idz] * x[idz]
                model.addConstr(con <= polytope.rhs[idy] + bigM * (1 - z[idx]),
                                name="pcon[" + str(idx) + "][" + str(idy) + "]")

                # model.update()
        model.addConstr(z[0] == 1)
        objective = LinExpr(0.0)
        for idx in range(n_z_vars):
            objective += z[idx]

        model.setObjective(objective, GRB.MAXIMIZE)

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.TimeLimit, 100.0)

        # model.write("testOnePolytope_model.lp")

        model.optimize()

        status = model.Status
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            print("Optimization stopped with status " + str(status))

        else:
            print("Bounds:\t" + str(int(model.objVal)) + "\t" + str(int(model.objBound)))

        for idx in range(n_z_vars):
            print("\t" + str(int(z[idx].getAttr(GRB.Attr.X))) + "\n")

        for idx in range(self.n_state_vars):
            print("\t" + str(float(x[idx].getAttr(GRB.Attr.X))) + "\n")

    def solve_4_both(self):
        start_time = time.time()
        model = Model("TestFarkasMIP")

        bigM = 100000.0
        lb = -10.0
        ub = 100.0

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

        # state variables
        alpha = []
        for idx in range(self.n_state_vars):
            temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="alpha[" + str(idx) + "]")
            alpha.append(temp_var)
        model.update()

        # binary decision variables
        z = []
        for idx in range(n_z_vars):
            temp_var = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="z[" + str(idx) + "]")
            z.append(temp_var)
        model.update()

        ida = 0  # to indexing A matrix
        for idx in range(n_z_vars):
            n1_constraints = n1_constraints_per_polytope[idx]
            for idy in range(n1_constraints):
                con = LinExpr(0.0)
                A1_row = A1_matrix[ida + idy]
                for idz in range(self.n_state_vars):
                    con += A1_row[idz] * alpha[idz]
                model.addConstr(con <= b1_array[ida+idy] + bigM * (1 - z[idx]),
                                name="Inst1con[" + str(idx) + "][" + str(idy) + "]")
            ida = ida + n1_constraints

        n_y_vars = 0
        y = []
        # y >= 0
        for idx in range(len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            for idy in range(n2_constraints):
                temp_var = model.addVar(lb=0.0, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="y[" + str(n_y_vars) + "]")
                y.append(temp_var)
                n_y_vars = n_y_vars + 1
        model.update()

        assert n_y_vars == len(A2_matrix)
        A2_matrix_t = np.transpose(A2_matrix)

        # A^T * y = 0
        for idx in range(len(A2_matrix_t)):
            A2_t_row = A2_matrix_t[idx]
            con = LinExpr(0.0)
            assert len(A2_t_row) == len(y)
            for idy in range(len(A2_t_row)):
                con += A2_t_row[idy] * y[idy]
            model.addConstr(con <= 0.0, name="Inst2con11[" + str(idx) + "]")
            model.addConstr(con >= 0.0, name="Inst2con12[" + str(idx) + "]")

        # Works for Quadcopter
        # epsilon = 0.000001

        # default
        epsilon = 0.0001
        # epsilon2 = 0.0005

        # b^T * y < 0
        b2_array_t = np.transpose(b2_array)
        assert n_y_vars == len(b2_array_t)
        con = LinExpr(0.0)
        for idy in range(len(b2_array_t)):
            con += b2_array_t[idy] * y[idy]
        model.addConstr(con <= -epsilon, name="Inst2con2")

        # y = 0 for z = 0, y > 0 for z = 1
        y_var_idx = n2_constraints_per_polytope[0]  # First entry corresponds to P2. The rest of them are for Q-set
        for idx in range(1, len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            # con = LinExpr(0.0)
            for idy in range(n2_constraints):
                # con += y[y_var_idx]
                model.addConstr(y[y_var_idx] <= bigM * z[idx], name="Inst2con31[" + str(idx) + "]")
                model.addConstr(y[y_var_idx] >= epsilon * z[idx], name="Inst2con32[" + str(idx) + "]")
                y_var_idx = y_var_idx + 1

        # ida = 0  # to indexing A matrix
        # for idx in range(n_z_vars):
        #     n2_constraints = n2_constraints_per_polytope[idx]
        #     for idy in range(n2_constraints):
        #         con = LinExpr(0.0)
        #         A2_row = A2_matrix[ida + idy]
        #         for idz in range(self.n_state_vars):
        #             con += A2_row[idz] * alpha[idz]
        #         model.addConstr(con <= b2_array[ida+idy] + bigM * (z[idx]),
        #                         name="Inst2con[" + str(idx) + "][" + str(idy) + "]")
        #     ida = ida + n2_constraints

        model.addConstr(z[0] <= 1, name="Inst2con41")
        model.addConstr(z[0] >= 1, name="Inst2con42")
        objective = LinExpr(0.0)
        for idx in range(n_z_vars):
            objective += z[idx]

        model.setObjective(objective, GRB.MAXIMIZE)

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.NodefileStart, 0.5)
        model.setParam(GRB.Param.TimeLimit, 100.0)
        model.setParam(GRB.Param.MIPGap, 1e-7)
        # model.setParam(GRB.Param.OptimalityTol, 1e-9)
        # model.setParam(GRB.Param.FeasibilityTol, 1e-9)
        # model.setParam(GRB.Param.IntFeasTol, 1e-9)

        # model.write("testFarkas_mip.lp")

        model.optimize()

        z_vals = []
        alpha_vals = []

        status = model.Status
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            print("Optimization stopped with status " + str(status))

        else:
            print("Bounds:\t" + str(int(model.objVal)) + "\t" + str(int(model.objBound)))

            for idx in range(n_z_vars):
                # print("\t" + str(int(z[idx].getAttr(GRB.Attr.X))) + "\n")
                z_vals.append(int(z[idx].getAttr(GRB.Attr.X)))

            for idx in range(self.n_state_vars):
                # print("\t" + str(float(alpha[idx].getAttr(GRB.Attr.X))) + "\n")
                alpha_vals.append(float(alpha[idx].getAttr(GRB.Attr.X)))

            # for idx in range(n_y_vars):
            #     print("\t" + str(float(y[idx].getAttr(GRB.Attr.X))) + "\n")
        print("Time taken by MIP: {}".format(str(time.time() - start_time)))

        return z_vals, alpha_vals

    def solve_4_both_redundant(self):
        start_time = time.time()
        model = Model("TestFarkasMIP")

        bigM = 100000.0
        lb = -10.0
        ub = 100.0

        n_z_vars = 1 + len(self.q_set) + 1

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

        A1_matrix = np.concatenate((A1_matrix, self.polytope2.con_matrix))
        b1_array = np.concatenate((b1_array, self.polytope2.rhs))
        n1_constraints_per_polytope.append(len(self.polytope2.rhs))
        # assert n_z_vars == len(n1_constraints_per_polytope)
        # assert len(A1_matrix) == np.sum(n1_constraints_per_polytope)

        # state variables
        alpha = []
        for idx in range(self.n_state_vars):
            temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="alpha[" + str(idx) + "]")
            alpha.append(temp_var)
        model.update()

        # binary decision variables
        z = []
        for idx in range(n_z_vars):
            temp_var = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="z[" + str(idx) + "]")
            z.append(temp_var)
        model.update()

        # Only for P2
        z_local = []
        for idy in range(len(self.polytope2.rhs)):
            temp_var = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY,
                                    name="z_local[" + str(idy) + "]")
            z_local.append(temp_var)

        ida = 0  # to indexing A matrix
        for idx in range(n_z_vars - 1):
            n1_constraints = n1_constraints_per_polytope[idx]
            for idy in range(n1_constraints):
                con = LinExpr(0.0)
                A1_row = A1_matrix[ida + idy]
                for idz in range(self.n_state_vars):
                    con += A1_row[idz] * alpha[idz]
                model.addConstr(con <= b1_array[ida+idy] + bigM * (1 - z[idx]),
                                name="Inst1con[" + str(idx) + "][" + str(idy) + "]")
            ida = ida + n1_constraints

        epsilon = 0.001
        zexpr = LinExpr(0.0)
        idx = n_z_vars - 1
        n1_constraints = n1_constraints_per_polytope[idx]
        for idy in range(n1_constraints):
            con = LinExpr(0.0)
            A1_row = A1_matrix[ida + idy]
            for idz in range(self.n_state_vars):
                con += A1_row[idz] * alpha[idz]
            model.addConstr(con <= b1_array[ida+idy] + bigM * (1 - z_local[idy]), name="Inst1con_local1[" + str(idy) + "]")
            model.addConstr(con >= b1_array[ida+idy] + epsilon - bigM * (z_local[idy]), name="Inst1con_local2[" + str(idy) + "]")
            model.addConstr(z[idx] <= z_local[idy], name="Inst1con_local3[" + str(idy) + "]")
            zexpr += z_local[idy]
        model.addConstr(zexpr <= z[idx] + n1_constraints - 1, name="Inst1con_local4")
        ida = ida + n1_constraints

        n_y_vars = 0
        y = []
        # y >= 0
        for idx in range(len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            for idy in range(n2_constraints):
                temp_var = model.addVar(lb=0.0, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="y[" + str(n_y_vars) + "]")
                y.append(temp_var)
                n_y_vars = n_y_vars + 1
        model.update()

        assert n_y_vars == len(A2_matrix)
        A2_matrix_t = np.transpose(A2_matrix)

        # A^T * y = 0
        for idx in range(len(A2_matrix_t)):
            A2_t_row = A2_matrix_t[idx]
            con = LinExpr(0.0)
            assert len(A2_t_row) == len(y)
            for idy in range(len(A2_t_row)):
                con += A2_t_row[idy] * y[idy]
            model.addConstr(con <= 0.0, name="Inst2con11[" + str(idx) + "]")
            model.addConstr(con >= 0.0, name="Inst2con12[" + str(idx) + "]")

        epsilon = 0.00001

        # b^T * y < 0
        b2_array_t = np.transpose(b2_array)
        assert n_y_vars == len(b2_array_t)
        con = LinExpr(0.0)
        for idy in range(len(b2_array_t)):
            con += b2_array_t[idy] * y[idy]
        model.addConstr(con <= -epsilon, name="Inst2con2")

        # y = 0 for z = 0, y > 0 for z = 1
        y_var_idx = n2_constraints_per_polytope[0]  # First entry corresponds to P2. The rest of them are for Q-set
        for idx in range(1, len(n2_constraints_per_polytope)):
            n2_constraints = n2_constraints_per_polytope[idx]
            # con = LinExpr(0.0)
            for idy in range(n2_constraints):
                # con += y[y_var_idx]
                model.addConstr(y[y_var_idx] <= bigM * z[idx], name="Inst2con31[" + str(idx) + "]")
                model.addConstr(y[y_var_idx] >= 0.0005 * z[idx], name="Inst2con32[" + str(idx) + "]")
                y_var_idx = y_var_idx + 1

        model.addConstr(z[0] <= 1, name="Inst2con41")
        model.addConstr(z[0] >= 1, name="Inst2con42")
        model.addConstr(z[n_z_vars-1] <= 0, name="Inst2con43")
        model.addConstr(z[n_z_vars-1] >= 0, name="Inst2con44")

        objective = LinExpr(0.0)
        for idx in range(n_z_vars):
            objective += z[idx]

        model.setObjective(objective, GRB.MAXIMIZE)

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.TimeLimit, 100.0)

        model.write("testFarkas_mip.lp")

        model.optimize()

        z_vals = []
        alpha_vals = []

        status = model.Status
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            print("Optimization stopped with status " + str(status))

        else:
            print("Bounds:\t" + str(int(model.objVal)) + "\t" + str(int(model.objBound)))

            for idx in range(n_z_vars):
                # print("\t" + str(int(z[idx].getAttr(GRB.Attr.X))) + "\n")
                z_vals.append(int(z[idx].getAttr(GRB.Attr.X)))

            for idx in range(self.n_state_vars):
                # print("\t" + str(float(alpha[idx].getAttr(GRB.Attr.X))) + "\n")
                alpha_vals.append(float(alpha[idx].getAttr(GRB.Attr.X)))

            # for idx in range(n_y_vars):
            #     print("\t" + str(float(y[idx].getAttr(GRB.Attr.X))) + "\n")
        print("Time taken by MIP: {}".format(str(time.time() - start_time)))

        return z_vals, alpha_vals
