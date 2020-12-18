import numpy as np
import scipy.linalg as spa
from hylaa.lpinstance import LpInstance
from gurobipy import *

'''
    An AH_polytope is an affine transformation of an H-polytope and is defined as:

    .. math::
        \mathbb{Q}=\{t+Tx  | x \in \mathbb{R}^p, H x \le h \}

    Attributes:
        * P: The underlying H-polytope :math:`P:\{x \in \mathbb{R}^p | Hx \le h\}`
        * T: :math:`\mathbb{R}^{n \times p}` matrix: linear transformation
        * t: :math:`\mathbb{R}^{n}` vector: translation 
'''


class AHPolytope:
    def __init__(self, H_polytope, T_matrix, t_vector):
        self.T_matrix = T_matrix
        self.t_vector = t_vector
        self.H_polytope = H_polytope

    def print(self):
        self.H_polytope.print()
        print(self.T_matrix, self.t_vector)
        return

    def check_inclusion(self, p1_ah_polytope):

        # ah_X \subseteq ah_Y

        P_x = p1_ah_polytope.H_polytope
        P_y = self.H_polytope
        H_x = P_x.H_matrix
        x_rhs = P_x.h_rhs

        H_y = P_y.H_matrix
        y_rhs = P_y.h_rhs

        y_rhs_flatten = y_rhs.flatten()

        x_rhs_flatten = x_rhs.flatten()

        x_bar = p1_ah_polytope.t_vector.flatten()
        X = p1_ah_polytope.T_matrix

        y_bar = self.t_vector.flatten()
        Y = self.T_matrix

        q_x, n_x = H_x.shape
        q_y, n_y = H_y.shape

        assert X.shape[1] == n_x
        assert Y.shape[1] == n_y
        print(n_x, n_y, q_x, q_y)

        model = Model("ahContainmentFeasibilityInst")
        lb = 0.0
        ub = 1000.0
        Gamma = []
        for idx in range(n_y):
            Gamma_p1 = []
            for idy in range(n_x):
                temp_var = model.addVar(lb=-ub, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="Gamma[" + str(idx) + "][" +
                                                                                          str(idy) + "]")
                Gamma_p1.append(temp_var)
            Gamma.append(Gamma_p1)
        model.update()

        Gamma_array = np.array(Gamma)

        Lambda = []
        for idx in range(q_y):
            Lambda_p1 = []
            for idy in range(q_x):
                temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="Lambda[" + str(idx) + "][" +
                                                                                          str(idy) + "]")
                Lambda_p1.append(temp_var)
            Lambda.append(Lambda_p1)
        model.update()

        Lambda_array = np.array(Lambda)

        beta = []
        for idx in range(n_y):
            temp_var = model.addVar(lb=-ub, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="beta[" + str(idx) + "]")
            beta.append(temp_var)

        model.update()

        Y_rows = Y.shape[0]
        # print(Gamma_array.shape)
        assert Y.shape[1] == Gamma_array.shape[0]
        for idx in range(Y_rows):
            Y_row = Y[idx]
            Gamma_cols = Gamma_array.shape[1]

            for idy in range(Gamma_cols):
                Gamma_col = Gamma_array[:, idy]
                # print(Gamma_col)
                lpi_con = LinExpr(0.0)
                for idz in range(Y.shape[1]):
                    lpi_con += Y_row[idz] * Gamma_col[idz]

                model.addConstr(lpi_con <= X[idx][idy],
                                name="first_con_leq[" + str(idx) + "_" + str(idy) + "]")
                model.addConstr(lpi_con >= X[idx][idy],
                                name="first_con_geq[" + str(idx) + "_" + str(idy) + "]")

        assert Lambda_array.shape[1] == H_x.shape[0]
        Lambda_rows = Lambda_array.shape[0]
        Lambda_Hx_matrix = []
        for idx in range(Lambda_rows):
            Lambda_row = Lambda_array[idx]

            Hx_cols = H_x.shape[1]

            Lambda_Hx_matrix_row = []
            for idy in range(Hx_cols):
                Hx_col = H_x[:, idy]
                lpi_con = LinExpr(0.0)
                for idz in range(Lambda_array.shape[1]):
                    lpi_con += Lambda_row[idz] * Hx_col[idz]
                Lambda_Hx_matrix_row.append(lpi_con)
            Lambda_Hx_matrix.append(Lambda_Hx_matrix_row)

        assert H_y.shape[1] == Gamma_array.shape[0]
        Hy_rows = H_y.shape[0]
        Hy_Gamma_matrix = []
        for idx in range(Hy_rows):
            Hy_row = H_y[idx]

            Gamma_cols = Gamma_array.shape[1]

            Hy_Gamma_matrix_row = []
            for idy in range(Gamma_cols):
                Gamma_col = Gamma_array[:, idy]
                lpi_con = LinExpr(0.0)
                for idz in range(H_y.shape[1]):
                    lpi_con += Hy_row[idz] * Gamma_col[idz]
                Hy_Gamma_matrix_row.append(lpi_con)
            Hy_Gamma_matrix.append(Hy_Gamma_matrix_row)

        Lambda_Hx_array = np.array(Lambda_Hx_matrix)
        Hy_Gamma_array = np.array(Hy_Gamma_matrix)
        assert Lambda_Hx_array.shape == Hy_Gamma_array.shape

        for idx in range(Lambda_Hx_array.shape[0]):
            for idy in range(Lambda_Hx_array.shape[1]):
                model.addConstr(Lambda_Hx_array[idx][idy] <= Hy_Gamma_array[idx][idy],
                                name="second_con_leq[" + str(idx) + "_" + str(idy) + "]")
                model.addConstr(Lambda_Hx_array[idx][idy] >= Hy_Gamma_array[idx][idy],
                                name="second_con_geq[" + str(idx) + "_" + str(idy) + "]")

        assert Y.shape[1] == len(beta)
        Y_beta_array = []
        for idx in range(Y.shape[0]):
            lpi_con = LinExpr(0.0)
            for idy in range(Y.shape[1]):
                lpi_con += Y[idx][idy] * beta[idy]
            Y_beta_array.append(lpi_con)

        assert len(Y_beta_array) == len(x_bar)
        assert len(Y_beta_array) == len(y_bar)
        for idx in range(len(Y_beta_array)):
            val = y_bar[idx] - x_bar[idx]
            model.addConstr(Y_beta_array[idx] <= val,
                            name="third_con_leq[" + str(idx) + "]")
            model.addConstr(Y_beta_array[idx] >= val,
                            name="third_con_geq[" + str(idx) + "]")

        assert H_y.shape[1] == len(beta)
        Hy_beta_array = []
        for idx in range(H_y.shape[0]):
            lpi_con = LinExpr(0.0)
            for idy in range(H_y.shape[1]):
                lpi_con += H_y[idx][idy] * beta[idy]
            Hy_beta_array.append(lpi_con)

        assert Lambda_array.shape[1] == len(x_rhs_flatten)
        Lambda_rhs_array = []
        for idx in range(Lambda_array.shape[0]):
            lpi_con = LinExpr(0.0)
            for idy in range(Lambda_array.shape[1]):
                lpi_con += Lambda_array[idx][idy] * x_rhs_flatten[idy]
            Lambda_rhs_array.append(lpi_con)

        assert len(Hy_beta_array) == len(Lambda_rhs_array)
        assert len(Hy_beta_array) == len(y_rhs_flatten)

        for idx in range(len(Hy_beta_array)):
            model.addConstr(Lambda_rhs_array[idx] <= y_rhs_flatten[idx] + Hy_beta_array[idx],
                            name="fourth_con_leq[" + str(idx) + "]")

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.TimeLimit, 100.0)

        # model.write("ah_contain_model.lp")

        model.optimize()

        status = model.Status
        ret_status = 0
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            # print("Optimization stopped with status " + str(status))
            ret_status = -1
        else:
            model.write("model_soln.sol")

        return ret_status

    # When we were converting alphas into std basis
    def check_inclusion_std_basis(self, P1_ah_polytope):

        # ah_X \subseteq ah_Y

        P_x = P1_ah_polytope.H_polytope
        P_y = self.H_polytope
        H_x = P_x.H_matrix
        x_rhs = P_x.h_rhs

        H_y = P_y.H_matrix
        y_rhs = P_y.h_rhs

        y_rhs_flatten = []
        for row in y_rhs:
            y_rhs_flatten = y_rhs_flatten + row.tolist()

        x_rhs_flatten = x_rhs.tolist()

        x_bar = P1_ah_polytope.t_vector.flatten()
        X = P1_ah_polytope.T_matrix

        y_bar = self.t_vector.flatten()
        Y = self.T_matrix

        q_x, n_x = H_x.shape
        q_y, n_y = H_y.shape

        assert X.shape[1] == n_x
        assert Y.shape[1] == n_y

        model = Model("ahContainmentFeasibilityInst")
        lb = 0.0
        ub = 1000.0
        Gamma = []
        for idx in range(n_y):
            Gamma_p1 = []
            for idy in range(n_x):
                temp_var = model.addVar(lb=-ub, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="Gamma[" + str(idx) + "][" +
                                                                                          str(idy) + "]")
                Gamma_p1.append(temp_var)
            Gamma.append(Gamma_p1)
        model.update()

        Gamma_array = np.array(Gamma)

        Lambda = []
        for idx in range(q_y):
            Lambda_p1 = []
            for idy in range(q_x):
                temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="Lambda[" + str(idx) + "][" +
                                                                                          str(idy) + "]")
                Lambda_p1.append(temp_var)
            Lambda.append(Lambda_p1)
        model.update()

        Lambda_array = np.array(Lambda)

        beta = []
        for idx in range(n_y):
            temp_var = model.addVar(lb=-ub, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="beta[" + str(idx) + "]")
            beta.append(temp_var)

        model.update()

        Y_rows = Y.shape[0]
        assert Y.shape[1] == Gamma_array.shape[0]
        for idx in range(Y_rows):
            Y_row = Y[idx]
            Gamma_cols = Gamma_array.shape[1]

            for idy in range(Gamma_cols):
                Gamma_col = Gamma_array[:, idy]
                lpi_con = LinExpr(0.0)
                for idz in range(Y.shape[1]):
                    lpi_con += Y_row[idz] * Gamma_col[idz]

                model.addConstr(lpi_con <= X[idx][idy],
                                name="first_con_leq[" + str(idx) + "_" + str(idy) + "]")
                model.addConstr(lpi_con >= X[idx][idy],
                                name="first_con_geq[" + str(idx) + "_" + str(idy) + "]")

        assert Lambda_array.shape[1] == H_x.shape[0]
        Lambda_rows = Lambda_array.shape[0]
        Lambda_Hx_matrix = []
        for idx in range(Lambda_rows):
            Lambda_row = Lambda_array[idx]

            Hx_cols = H_x.shape[1]

            Lambda_Hx_matrix_row = []
            for idy in range(Hx_cols):
                Hx_col = H_x[:, idy]
                lpi_con = LinExpr(0.0)
                for idz in range(Lambda_array.shape[1]):
                    lpi_con += Lambda_row[idz] * Hx_col[idz]
                Lambda_Hx_matrix_row.append(lpi_con)
            Lambda_Hx_matrix.append(Lambda_Hx_matrix_row)

        assert H_y.shape[1] == Gamma_array.shape[0]
        Hy_rows = H_y.shape[0]
        Hy_Gamma_matrix = []
        for idx in range(Hy_rows):
            Hy_row = H_y[idx]

            Gamma_cols = Gamma_array.shape[1]

            Hy_Gamma_matrix_row = []
            for idy in range(Gamma_cols):
                Gamma_col = Gamma_array[:, idy]
                lpi_con = LinExpr(0.0)
                for idz in range(H_y.shape[1]):
                    lpi_con += Hy_row[idz] * Gamma_col[idz]
                Hy_Gamma_matrix_row.append(lpi_con)
            Hy_Gamma_matrix.append(Hy_Gamma_matrix_row)

        Lambda_Hx_array = np.array(Lambda_Hx_matrix)
        Hy_Gamma_array = np.array(Hy_Gamma_matrix)
        assert Lambda_Hx_array.shape == Hy_Gamma_array.shape

        for idx in range(Lambda_Hx_array.shape[0]):
            for idy in range(Lambda_Hx_array.shape[1]):
                model.addConstr(Lambda_Hx_array[idx][idy] <= Hy_Gamma_array[idx][idy],
                                name="second_con_leq[" + str(idx) + "_" + str(idy) + "]")
                model.addConstr(Lambda_Hx_array[idx][idy] >= Hy_Gamma_array[idx][idy],
                                name="second_con_geq[" + str(idx) + "_" + str(idy) + "]")

        assert Y.shape[1] == len(beta)
        Y_beta_array = []
        for idx in range(Y.shape[0]):
            lpi_con = LinExpr(0.0)
            for idy in range(Y.shape[1]):
                lpi_con += Y[idx][idy] * beta[idy]
            Y_beta_array.append(lpi_con)

        assert len(Y_beta_array) == len(x_bar)
        assert len(Y_beta_array) == len(y_bar)
        for idx in range(len(Y_beta_array)):
            val = y_bar[idx] - x_bar[idx]
            model.addConstr(Y_beta_array[idx] <= val,
                            name="third_con_leq[" + str(idx) + "]")
            model.addConstr(Y_beta_array[idx] >= val,
                            name="third_con_geq[" + str(idx) + "]")

        assert H_y.shape[1] == len(beta)
        Hy_beta_array = []
        for idx in range(H_y.shape[0]):
            lpi_con = LinExpr(0.0)
            for idy in range(H_y.shape[1]):
                lpi_con += H_y[idx][idy] * beta[idy]
            Hy_beta_array.append(lpi_con)

        assert Lambda_array.shape[1] == len(x_rhs_flatten)
        Lambda_rhs_array = []
        for idx in range(Lambda_array.shape[0]):
            lpi_con = LinExpr(0.0)
            for idy in range(Lambda_array.shape[1]):
                lpi_con += Lambda_array[idx][idy] * x_rhs_flatten[idy]
            Lambda_rhs_array.append(lpi_con)

        assert len(Hy_beta_array) == len(Lambda_rhs_array)
        assert len(Hy_beta_array) == len(y_rhs_flatten)

        for idx in range(len(Hy_beta_array)):
            model.addConstr(Lambda_rhs_array[idx] <= y_rhs_flatten[idx] + Hy_beta_array[idx],
                            name="fourth_con_leq[" + str(idx) + "]")

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.TimeLimit, 100.0)

        # model.write("ah_contain_model.lp")

        model.optimize()

        status = model.Status
        ret_status = 0
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            # print("Optimization stopped with status " + str(status))
            ret_status = -1
        else:
            model.write("model_soln.sol")

        return ret_status


class HPolytope:
    def __init__(self, H_matrix, h_rhs, con_types=None):
        self.H_matrix = H_matrix
        self.h_rhs = h_rhs
        if con_types is None:
            con_types = np.empty(h_rhs.shape)
            con_types.fill(3)
        self.con_types = con_types

    def print(self):
        print("Print polytope with " + str(self.H_matrix.shape[1]) + " variables and " + str(len(self.con_types))
              + " constraints...")
        print(self.H_matrix)
        print(self.h_rhs)
        print(self.con_types)
        return

    def convert_to_std_h_polytope(self):
        con_matrix = []
        rhs = []
        con_types = []
        for idx in range(len(self.con_types)):
            con_matrix_row = self.H_matrix[idx]
            con_matrix.append(con_matrix_row)
            rhs.append(self.h_rhs[idx])
            con_types.append(3)
            if self.con_types[idx] == 5:
                con_matrix.append([val * -1 + 0.0 for val in con_matrix_row])
                rhs.append(self.h_rhs[idx] * -1)
                con_types.append(3)
        self.H_matrix = np.array(con_matrix)
        self.con_types = np.array(con_types)
        self.h_rhs = np.array(rhs)
        # print(np.array(self.con_matrix))
        # print(self.rhs)
        # print(self.con_types)

    def convert_to_AH_polytope(self, T_matrix=None, t_vector=None):
        # n = self.H_matrix.shape[1]
        if T_matrix is not None:
            n = T_matrix.shape[0]
        else:
            n = self.H_matrix.shape[1]
        if T_matrix is None:
            T_matrix = np.eye(n)
        if t_vector is None:
            t_vector = np.zeros((n, 1))
        AH_polytope = AHPolytope(H_polytope=HPolytope(H_matrix=self.H_matrix, h_rhs=self.h_rhs, con_types=self.con_types),
                                 T_matrix=T_matrix,
                                 t_vector=t_vector)
        return AH_polytope


def convert_lpi_to_ah_polytope(p1_lpi, dims):
    Y_matrix = np.eye(dims)
    t_vector = np.zeros((dims, 1))
    con_matrix = p1_lpi.get_full_constraints().toarray()
    lpi_rhs = p1_lpi.get_rhs().tolist()
    lpi_types = p1_lpi.get_types()
    dim1 = con_matrix.shape[0]
    # H_matrix = np.zeros((dim1-dims, dims), dtype=float)
    H_matrix = con_matrix[dims:dim1, 0:dims]
    h_rhs = lpi_rhs[dims:dim1]
    con_types = lpi_types[dims:dim1]
    cur_h_polytope = HPolytope(H_matrix=np.array(H_matrix), h_rhs=np.array(h_rhs), con_types=np.array(con_types))
    cur_h_polytope.convert_to_std_h_polytope()
    cur_ah_polytope = cur_h_polytope.convert_to_AH_polytope(T_matrix=Y_matrix, t_vector=t_vector)
    # cur_ah_polytope.print()
    return cur_ah_polytope


def convert_lpi_to_ah_polytope_std_basis(P1_lpi):
    P1_ah_polytope = P1_lpi
    if isinstance(P1_lpi, LpInstance):
        con_matrix = P1_lpi.get_full_constraints().toarray()

        dims = 3
        std_basis_init_pred_h_matrix = con_matrix[dims:dims * dims, 0:dims]
        basis_matrix = con_matrix[0:dims, 0:dims]
        auto_h_matrix = np.matmul(std_basis_init_pred_h_matrix, np.linalg.inv(basis_matrix))
        h_rhs = P1_lpi.get_rhs().tolist()[dims:dims * dims]
        con_types = P1_lpi.get_types()[dims:dims * dims]

        # h_rhs = P1_lpi.get_rhs().tolist()
        # con_types = P1_lpi.get_types()
        # P1_h_polytope = HPolytope(H_matrix=con_matrix, h_rhs=np.array(h_rhs), con_types=np.array(con_types))
        P1_h_polytope = HPolytope(H_matrix=auto_h_matrix, h_rhs=h_rhs, con_types=con_types)
        P1_h_polytope.convert_to_std_h_polytope()
        # P1_h_polytope.print()
        P1_ah_polytope = P1_h_polytope.convert_to_AH_polytope()

    return P1_ah_polytope


def minkowski_sum(P1_polytope, P2_polytope):
    if isinstance(P1_polytope, AHPolytope):
        P1_AHpolytope = P1_polytope
    elif isinstance(P1_polytope, HPolytope):
        P1_AHpolytope = P1_polytope.convert_to_AH_polytope()
    else:
        print("Incorrect polytope type")
        return

    # P1_AHpolytope.print()
    if isinstance(P2_polytope, AHPolytope):
        P2_AHpolytope = P2_polytope
    elif isinstance(P2_polytope, HPolytope):
        P2_AHpolytope = P2_polytope.convert_to_AH_polytope()
    else:
        print("Incorrect polytope type")
        return

    # P2_AHpolytope.print()
    T_matrix = np.hstack((P1_AHpolytope.T_matrix, P2_AHpolytope.T_matrix))
    t_vector = P2_AHpolytope.t_vector + P2_AHpolytope.t_vector
    H_matrix = spa.block_diag(*[P1_AHpolytope.H_polytope.H_matrix, P2_AHpolytope.H_polytope.H_matrix])
    h_rhs = np.vstack((P1_AHpolytope.H_polytope.h_rhs, P2_AHpolytope.H_polytope.h_rhs))
    new_Hpolytope = HPolytope(H_matrix=H_matrix, h_rhs=h_rhs)
    return AHPolytope(H_polytope=new_Hpolytope, T_matrix=T_matrix, t_vector=t_vector)


