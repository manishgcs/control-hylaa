from gurobipy import *
import numpy as np
from hylaa.lpinstance import LpInstance
from hylaa.lpinstance import LpInstance


def check_poly_contain_efficient(P1_lpi, P2_equations):
    assert P1_lpi is not None

    P2_equations = P2_equations.tolist()
    con_matrix = []
    rhs = []
    con_types = []
    for idx in range(len(P2_equations)):
        P2_equation = P2_equations[idx]
        rhs.append(-1 * P2_equation.pop())
        lhs = P2_equation
        con_matrix.append(lhs)
        con_types.append(3)

    # print(con_matrix)
    # print(rhs)
    # print(con_types)

    polyt = Polytope(con_matrix=con_matrix, rhs=rhs, con_types=con_types)
    polyt.convert_to_std_h_polytope()

    # polyt.polytopePrint()

    P1_poly = None
    # if isinstance(P1_lpi, LpInstance):
    #     P1_poly = Polytope(con_matrix=P1_lpi.get_full_constraints().toarray().tolist(), rhs=P1_lpi.get_rhs().tolist(),
    #                    con_types=P1_lpi.get_types())
    #     P1_poly.polytopePrint()

    if isinstance(P1_lpi, Polytope):
        P1_poly = P1_lpi

    assert P1_poly is not None
    P1_poly.convert_to_std_h_polytope()
    # P1_poly.polytopePrint()
    # P1_poly.project_on_polyt_dims(polyt)
    # P1_poly.polytopePrint()
    p_contain_inst = PContainFeasibilityInstance(poly1=P1_poly, poly2=polyt)
    p_contain_inst.solve()


def check_poly_contain_efficient_w_proj(P1_lpi, P2_lpi):
    assert P1_lpi is not None

    polyt = Polytope(con_matrix=P2_lpi.get_full_constraints().toarray().tolist(), rhs=P2_lpi.get_rhs().tolist(),
                     con_types=P2_lpi.get_types())
    # polyt.polytopePrint()
    # print(polyt.con_types)
    polyt.convert_to_std_h_polytope()

    # polyt.polytopePrint()

    P1_poly = Polytope(con_matrix=P1_lpi.get_full_constraints().toarray().tolist(), rhs=P1_lpi.get_rhs().tolist(),
                       con_types=P1_lpi.get_types())
    # P1_poly.polytopePrint()

    # P1_poly_con_matrix = [[0, 0, 0, -1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 1, 0],
    # [0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 1]]
    # P1_poly_rhs = [-0.4, 5, 0.2, 0.5, 0.2, 0.5]
    # P1_poly_types = [3, 3, 3, 3, 3, 3]
    # print(P1_poly_con_matrix)
    # P1_poly = Polytope(con_matrix=P1_poly_con_matrix, rhs=P1_poly_rhs, con_types=P1_poly_types)
    P1_poly.convert_to_std_h_polytope()
    # P1_poly.polytopePrint()
    P1_poly.project_on_polyt_dims(polyt)
    # P1_poly.polytopePrint()
    p_contain_inst = PContainFeasibilityInstance(poly1=P1_poly, poly2=polyt)
    p_contain_inst.solve()


def check_poly_contain_brute_force(P1_box, P2_lpi):
    # print("*********** in lpplot *************")
    # print(self.lpi.get_full_constraints().toarray())
    # print(self.lpi.get_rhs())
    # print(self.lpi.get_types())
    # print(lpi.get_full_constraints().toarray().shape, len(lpi.get_rhs().tolist()), len(lpi.get_types()))

    P1_tmpc_model = P1_box

    dim_1 = P1_tmpc_model[0]
    vertices = []
    for idx in range(2):
        dim_2 = P1_tmpc_model[1]
        for idy in range(2):
            dim_3 = P1_tmpc_model[2]
            for idz in range(2):
                vertex = [dim_1[idx], dim_2[idy], dim_3[idz]]
                vertices.append(vertex)

    # vertices = [[0.6, -0.2, -0.2], [0.6, -0.2, 0.5], [0.6, 0.5, -0.2], [0.6, 0.5, 0.5],
    #             [5.0, -0.2, -0.2], [5.0, -0.2, 0.5], [5.0, 0.5, -0.2], [5.0, 0.5, 0.5]]
    # print(vertices)
    # polyt = Polytope(con_matrix=self.lpi.get_full_constraints(), rhs=self.lpi.get_rhs(),
    #                  con_types=self.lpi.get_types())
    polyt = Polytope(con_matrix=P2_lpi.get_full_constraints().toarray().tolist(), rhs=P2_lpi.get_rhs().tolist(),
                     con_types=P2_lpi.get_types())
    polyt.polytopePrint()

    f_instance = VFeasibilityInstance(polytope=polyt)
    status = 0
    for idx in range(len(vertices)):
        vertex = vertices[idx]
        status = f_instance.solve(vertex=vertex)
        if status == -1:
            status = idx + 1
            break
    if status > 0:
        print("P1 is not contained in P2 for vertex " + str(status))
    else:
        print("P1 is contained in P2")


class Polytope:
    def __init__(self, con_matrix, rhs, con_types):
        # self.con_matrix = con_matrix.toarray().tolist()
        # self.rhs = rhs.tolist()
        self.con_matrix = con_matrix
        self.rhs = rhs
        self.con_types = con_types
        self.n_variables = len(self.con_matrix[0])

    def polytopePrint(self):
        print("Print polytope with " + str(self.n_variables) + " variables and " + str(len(self.con_types))
              + " constraints...")
        print(np.array(self.con_matrix), np.array(self.rhs), np.array(self.con_types))

    def get_dimensions(self):
        return self.n_variables

    def get_con_types(self):
        return self.con_types

    def get_con_matrix(self):
        return self.con_matrix

    def get_rhs(self):
        return self.rhs

    def convert_to_std_h_polytope(self):
        # print(np.array(self.con_matrix))
        # print(self.rhs)
        # print(self.con_types)
        con_matrix = []
        rhs = []
        con_types = []
        for idx in range(len(self.con_types)):
            con_matrix_row = self.con_matrix[idx]
            con_matrix.append(con_matrix_row)
            rhs.append(self.rhs[idx])
            con_types.append(3)
            if self.con_types[idx] == 5:
                con_matrix.append([val * -1 + 0.0 for val in con_matrix_row])
                rhs.append(self.rhs[idx] * -1)
                con_types.append(3)
        self.con_matrix = con_matrix
        self.con_types = con_types
        self.rhs = rhs
        # print(np.array(self.con_matrix))
        # print(self.rhs)
        # print(self.con_types)

    def project_on_polyt_dims(self, polyt):
        polyt_dims = polyt.n_variables
        assert self.n_variables <= polyt_dims
        array_to_stack = np.zeros((len(self.con_types), polyt_dims - self.n_variables))
        print(np.array(self.con_matrix).shape, array_to_stack.shape)
        print(np.array(polyt.con_matrix).shape)
        new_array = np.column_stack((np.array(self.con_matrix), array_to_stack))
        print(new_array.shape)
        self.con_matrix = new_array.tolist()
        self.n_variables = new_array.shape[1]


class PContainFeasibilityInstance:
    def __init__(self, benchmark=None, poly1=None, poly2=None):
        self.benchmark = benchmark
        self.polytope1 = poly1
        self.polytope2 = poly2

    def solve(self):
        model = Model("pContainFeasibilityInst")
        lb = 0.0
        ub = 1000.0

        p1_con_matrix = self.polytope1.get_con_matrix()
        p1_con_types = self.polytope1.get_con_types()
        p1_rhs = self.polytope1.get_rhs()

        p2_con_matrix = self.polytope2.get_con_matrix()
        p2_con_types = self.polytope2.get_con_types()
        p2_rhs = self.polytope2.get_rhs()

        n_constraints_p1 = len(p1_con_types)
        n_constraints_p2 = len(p2_con_types)

        print(n_constraints_p2, n_constraints_p1)
        x = []
        for idx in range(n_constraints_p2):
            x_p1 = []
            for idz in range(n_constraints_p1):
                temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="x[" + str(idx) + "][" +
                                                                                          str(idz) + "]")
                x_p1.append(temp_var)
            x.append(x_p1)
        model.update()

        # for idx in range(n_constraints_p2):
        #     p1_con_matrix_tr = np.array(p1_con_matrix).T.tolist()
        #     p2_con_matrix_row = p2_con_matrix[idx]
        #     assert len(p2_con_matrix_row) == len(p1_con_matrix_tr)
        #     for idy in range(len(p1_con_matrix_tr)):
        #         p1_con_matrix_tr_row = p1_con_matrix_tr[idy]
        #         lpi_con = LinExpr(0.0)
        #         assert len(p1_con_matrix_tr_row) == n_constraints_p1
        #         # print(p1_con_matrix_tr_row)
        #         if not np.all((np.array(p1_con_matrix_tr_row) == 0.0)):
        #             for idz in range(n_constraints_p1):
        #                 lpi_con += p1_con_matrix_tr_row[idz] * x[idx][idz]
        #             model.addConstr(lpi_con <= p2_con_matrix_row[idy],
        #                             name="lpi_con_leq[" + str(idx) + "_" + str(idy) + "]")
        #             model.addConstr(lpi_con >= p2_con_matrix_row[idy],
        #                             name="lpi_con_geq[" + str(idx) + "_" + str(idy) + "]")

        for idx in range(n_constraints_p2):
            p2_con_matrix_row = p2_con_matrix[idx]
            p1_con_matrix_array = np.array(p1_con_matrix)
            # print(p1_con_matrix_array.shape)
            p1_con_n_rows, p1_con_n_cols = p1_con_matrix_array.shape
            assert len(p2_con_matrix_row) == p1_con_n_cols
            for idy in range(p1_con_n_cols):
                p1_con_matrix_array_col = p1_con_matrix_array[:, idy]
                lpi_con = LinExpr(0.0)
                assert len(p1_con_matrix_array_col) == n_constraints_p1 == p1_con_n_rows
                # print(p1_con_matrix_tr_row)
                if not np.all((p1_con_matrix_array_col == 0.0)):
                    for idz in range(n_constraints_p1):
                        lpi_con += p1_con_matrix_array_col[idz] * x[idx][idz]
                    model.addConstr(lpi_con <= p2_con_matrix_row[idy],
                                    name="lpi_con_leq[" + str(idx) + "_" + str(idy) + "]")
                    model.addConstr(lpi_con >= p2_con_matrix_row[idy],
                                    name="lpi_con_geq[" + str(idx) + "_" + str(idy) + "]")

        for idx in range(n_constraints_p2):
            rhs_con = LinExpr(0.0)
            for idz in range(n_constraints_p1):
                rhs_con += p1_rhs[idz] * x[idx][idz]
            model.addConstr(rhs_con <= p2_rhs[idx], name="rhs_con_leq[" + str(idx) + "]")

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.TimeLimit, 100.0)

        model.write("p_contain_model.lp")

        model.optimize()

        # obj_con = LinExpr(0.0)
        # for idx in range(1, 2):
        #     for idz in range(n_constraints_p1):
        #         obj_con += p1_rhs[idz] * x[idx][idz]

        # model.setObjective(obj_con, GRB.MINIMIZE)

        status = model.Status
        ret_status = 0
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            # print("Optimization stopped with status " + str(status))
            ret_status = -1
        else:
            model.write("model_soln.sol")
        # else:
        #     for idx in range(n_constraints_p2):
        #         lhs_val = float(0.0)
        #         for idz in range(n_constraints_p1):
        #             lhs_val += p1_rhs[idz] * float(x[idx][idz].X)
        #         print(" **** LHS val " + str(lhs_val) + " ****** rhs " + str(p2_rhs[idx]))

        print("******* Return status *******" + str(ret_status))


class VFeasibilityInstance:
    def __init__(self, benchmark=None, polytope=None):
        self.benchmark = benchmark
        self.polytope = polytope
        self.n_variables = polytope.get_dimensions()

    def solve(self, vertex=None):
        model = Model("vFeasibilityInst")
        lb = -50.0
        ub = 50.0

        x = []
        for idx in range(self.n_variables):
            temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="x[" + str(idx) + "]")
            x.append(temp_var)
        model.update()

        con_matrix = self.polytope.con_matrix
        rhs = self.polytope.rhs
        types = self.polytope.con_types
        for idx in range(len(con_matrix)):
            con_matrix_row = con_matrix[idx]
            r_con = LinExpr(0.0)
            for idy in range(len(con_matrix_row)):
                r_con += con_matrix_row[idy] * x[idy]
            model.addConstr(r_con <= rhs[idx], name="r_con[" + str(idx) + "_" + str(0) + "]")
            if types[idx] == 5:
                model.addConstr(r_con >= rhs[idx], name="r_con[" + str(idx) + "_" + str(1) + "]")

        dimensions = len(vertex)

        for idx in range(dimensions):
            v_con = LinExpr(0.0)
            v_con += x[dimensions+idx]
            model.addConstr(v_con <= vertex[idx], name="v_con[" + str(idx) + "_" + str(0) + "]")
            model.addConstr(v_con >= vertex[idx], name="v_con[" + str(idx) + "_" + str(1) + "]")

        # objective = LinExpr(0.0)
        # objective += x[2]
        #
        # model.setObjective(objective, GRB.MAXIMIZE)

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.TimeLimit, 100.0)

        model.write("v_feasibility_model.lp")

        model.optimize()

        status = model.Status
        ret_status = 0
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            # print("Optimization stopped with status " + str(status))
            ret_status = -1
        # else:
        #     # print("Bounds:\t" + str(int(model.objVal)) + "\t" + str(int(model.objBound)))

        # for idx in range(self.n_variables):
        #     print("\t" + str(int(x[idx].getAttr(GRB.Attr.X))) + "\n")
        #
        # for idx in range(self.n_variables):
        #     print("\t" + str(float(x[idx].getAttr(GRB.Attr.X))) + "\n")

        return ret_status


