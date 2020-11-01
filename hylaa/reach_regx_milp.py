import ast
import numpy as np
from gurobipy import *


class Polytope:
    def __init__(self, n_constraints, con_matrix, rhs):
        self.n_constraints = n_constraints
        self.con_matrix = con_matrix
        self.rhs = rhs

    def polytopePrint(self):
        print("Print polytope with " + str(len(self.n_constraints)) + " constraints...")


class RegexInstance:
    def __init__(self, benchmark, n_variables, n_timesteps, filename=None):
        self.n_variables = n_variables
        self.n_timesteps = n_timesteps
        self.filename = filename
        self.benchmark = benchmark
        self.polytopes = []

    def addPolytope(self, polytope):
        self.polytopes.append(polytope)

    def solve(self, reg_expr):
        model = Model("RegexInstance")

        epsilon = 0.000001
        bigM = 1000.0
        lb = -100000.0
        ub = 1.0
        if self.benchmark == "Ball":
            bigM = 1000.0
            lb = -1000.0
            ub = 10.0
        elif self.benchmark == "Oscillator":
            bigM = 10000.0
            lb = -100000.0
            ub = 1.0
        elif self.benchmark == "Tanks":
            bigM = 100000.0
            lb = -1000000.0
            ub = 10.0
        elif self.benchmark == "Buck":
            bigM = 1000.0
            lb = -1000000.0
            ub = 1.0
        elif self.benchmark == "Filtered":
            bigM = 1000.0
            lb = -100000.0
            ub = 100.0
        elif self.benchmark == "ISS":
            bigM = 10.0
            lb = -1000000.0
            ub = 100.0
        elif self.benchmark == "Particle":
            bigM = 100.0
            lb = -100000.0
            ub = 10.0

        print("big M is %f" % bigM)

        x = []
        for idx in range(self.n_variables):
            temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="x[" + str(idx) + "]")
            x.append(temp_var)
        model.update()

        z = []
        for idx in range(self.n_timesteps):
            temp_var = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="z[" + str(idx) + "]")
            z.append(temp_var)
        model.update()

        z_local = []
        for idx in range(self.n_timesteps):
            z_t = []
            polytope = self.polytopes[idx]
            for idy in range(polytope.n_constraints):
                temp_var = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="z_local[" + str(idx) + "][" + str(idy) + "]")
                z_t.append(temp_var)
            z_local.append(z_t)

        for idx in range(self.n_timesteps):
            zexpr = LinExpr(0.0)
            polytope = self.polytopes[idx]
            for idy in range(polytope.n_constraints):
                con = LinExpr(0.0)
                for idz in range(self.n_variables):
                    con += polytope.con_matrix[idy][idz] * x[idz]
                model.addConstr(con <= polytope.rhs[idy] + bigM * (1 - z_local[idx][idy]), name="pcon_1[" + str(idx) + "][" + str(idy) + "]")
                model.addConstr(con >= polytope.rhs[idy] + epsilon - bigM *(z_local[idx][idy]), name="pcon_2[" + str(idx) + "][" + str(idy) + "]")
                model.addConstr(z[idx] <= z_local[idx][idy], name="zcon_1[" + str(idx) + "][" + str(idy) + "]")
                zexpr += z_local[idx][idy]
            model.addConstr(zexpr <= z[idx] + polytope.n_constraints -1, name="zcon[" + str(idx) + "]")
            if reg_expr[idx] == '1':
                model.addConstr(z[idx] >= 1)
                model.addConstr(z[idx] <= 1)
            elif reg_expr[idx] == '0':
                model.addConstr(z[idx] >= 0)
                model.addConstr(z[idx] <= 0)

        objective = LinExpr(0.0)
        for idx in range(self.n_timesteps):
            objective += z[idx]

        model.setObjective(objective, GRB.MAXIMIZE)

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.TimeLimit, 100.0)

        model.write("regex_model.lp")

        model.optimize()

        status = model.Status
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            print("Optimization stopped with status " + str(status))

        else:
            print("Bounds:\t" + str(int(model.objVal)) + "\t" + str(int(model.objBound)))

        for idx in range(self.n_timesteps):
            print("\t" + str(int(z[idx].getAttr(GRB.Attr.X))) + "\n")

        for idx in range(self.n_variables):
            print("\t" + str(float(x[idx].getAttr(GRB.Attr.X))) + "\n")


class ReachabilityInstance:
    def __init__(self, benchmark, n_variables, n_timesteps, filename=None):
        self.n_variables = n_variables
        self.n_timesteps = n_timesteps
        self.filename = filename
        self.benchmark = benchmark
        self.polytopes = []

    def addPolytope(self, polytope):
        self.polytopes.append(polytope)

    def read(self):
        # print "Filename is: " + self.filename
        f = open(self.filename, "r")
        lines = f.readlines()
        self.n_variables = int(lines[0])
        self.n_timesteps = int(lines[1])
        l_idx = 2

        for idt in range(self.n_timesteps):
            n_constraints = int(lines[l_idx])
            l_idx = l_idx + 1
            con_matrix = np.zeros(n_constraints, self.n_variables)
            rhs = np.zeros(n_constraints)
            for idx in range(n_constraints):
                con_data_str = lines[l_idx]
                l_idx = l_idx + 1
                con_data = ast.literal_eval(con_data_str)
                n_vars_in_con = int(con_data[0])
                for idy in range(n_vars_in_con):
                    var = int(con_data[1+idy*2])
                    coeff = float(con_data[2+idy*2])
                    con_matrix[idx][var-1] = coeff
                rhs[idx] = float(con_data[len(con_data)-1])
            polytope = Polytope(n_constraints. con_matrix, rhs)
            self.polytopes.append(polytope)

    def solve(self):
        model = Model("ReachabilityInstance")

        bigM = 1000.0
        lb = -100000.0
        ub = 1.0
        if self.benchmark == "Ball":
            bigM = 1000.0
            lb = -1000.0
            ub = 10.0
        elif self.benchmark == "Oscillator":
            bigM = 10000.0
            lb = -100000.0
            ub = 1.0
        elif self.benchmark == "Tanks":
            bigM = 100000.0
            lb = -1000000.0
            ub = 10.0
        elif self.benchmark == "Buck":
            bigM = 1000.0
            lb = -1000000.0
            ub = 1.0
        elif self.benchmark == "Filtered":
            bigM = 1000.0
            lb = -100000.0
            ub = 100.0
        elif self.benchmark == "ISS":
            bigM = 10.0
            lb = -1000000.0
            ub = 100.0
        elif self.benchmark == "Particle":
            bigM = 100.0
            lb = -100000.0
            ub = 10.0

        print("big M is %f" % bigM)

        x = []
        for idx in range(self.n_variables):
            temp_var = model.addVar(lb=lb, ub=ub, obj=0.0, vtype=GRB.CONTINUOUS, name="x[" + str(idx) + "]")
            x.append(temp_var)
        model.update()

        z = []
        for idx in range(self.n_timesteps):
            temp_var = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="z[" + str(idx) + "]")
            z.append(temp_var)
        model.update()

        for idx in range(self.n_timesteps):
            polytope = self.polytopes[idx]
            for idy in range(polytope.n_constraints):
                con = LinExpr(0.0)
                for idz in range(self.n_variables):
                    con += polytope.con_matrix[idy][idz] * x[idz]
                model.addConstr(con <= polytope.rhs[idy] + bigM * (1 - z[idx]), name="pcon[" + str(idx) + "][" + str(idy) + "]")

        # model.update()
        objective = LinExpr(0.0)
        for idx in range(self.n_timesteps):
            objective += z[idx]

        model.setObjective(objective, GRB.MAXIMIZE)

        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.TimeLimit, 100.0)

        model.write("reachability_model.lp")

        model.optimize()

        status = model.Status
        if (status != GRB.OPTIMAL) or (status == GRB.INF_OR_UNBD) or (status == GRB.INFEASIBLE) or (
                status == GRB.UNBOUNDED):
            print("Optimization stopped with status " + str(status))

        else:
            print("Bounds:\t" + str(int(model.objVal)) + "\t" + str(int(model.objBound)))

        for idx in range(self.n_timesteps):
            print("\t" + str(int(z[idx].getAttr(GRB.Attr.X))) + "\n")

        for idx in range(self.n_variables):
            print("\t" + str(float(x[idx].getAttr(GRB.Attr.X))) + "\n")
