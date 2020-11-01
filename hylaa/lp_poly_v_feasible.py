from gurobipy import *


class Polytope:
    def __init__(self, con_matrix, rhs, con_types):
        self.con_matrix = con_matrix.toarray().tolist()
        self.rhs = rhs.tolist()
        self.con_types = con_types
        self.n_variables = len(self.con_matrix[0])

    def polytopePrint(self):
        print("Print polytope with " + str(self.n_variables) + " variables and " + str(len(self.con_types))
              + " constraints...")


class VFeasibilityInstance:
    def __init__(self, benchmark=None, polytope=None):
        self.benchmark = benchmark
        self.polytope = polytope
        self.n_variables = self.polytope.n_variables

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

        model.write("regex_model.lp")

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


