# this file is prepared for project 026
# Created by iboxl

from gurobipy import Model, GRB, quicksum
import gurobipy as gp
from gurobipy import *
model = Model("matrix_multiplication")
model.setParam('NonConvex', 2)
model.setParam('Threads', 80)
model.setParam("FeasibilityTol", 1e-9)
model.setParam("IntFeasTol", 1e-9)
model.setParam('MIPFocus', 1)
model.setParam("ZeroObjNodes",100)
model.setParam("RINS",1000)

M = model.addVar(lb = 5, ub = 1200, vtype=GRB.INTEGER)
N = model.addVar(lb = 5, ub = 1200, vtype=GRB.INTEGER)
K = model.addVar(lb = 5, ub = 1200, vtype=GRB.INTEGER)
m = model.addVar(lb = 5, vtype=GRB.INTEGER)
n = model.addVar(lb = 5, vtype=GRB.INTEGER)
k = model.addVar(lb = 5, vtype=GRB.INTEGER)
model.addConstr( m * 2 <= M )
model.addConstr( n * 2 <= N )
model.addConstr( k * 2 <= K )
model.addConstr( m * n == 128)

v1 = model.addVar(vtype=GRB.INTEGER)
v2 = model.addVar(vtype=GRB.INTEGER)
v3 = model.addVar(vtype=GRB.INTEGER)
v4 = model.addVar(vtype=GRB.INTEGER)
v5 = model.addVar(vtype=GRB.INTEGER)
v6 = model.addVar(vtype=GRB.INTEGER)

v7 = model.addVar(lb=0, vtype=GRB.INTEGER)


model.addConstr(v1 == M*N - m*n)
model.addConstr(v2 * k == K - k)
model.addConstr(v3 * m == M - m)
model.addConstr(v4 == K*N)
model.addConstr(v5 * n == N - n)
model.addConstr(v6 == K*M)

model.addConstr(4*v1*v2 + v7 <= v3*v4 + v5*v6)

model.update()

model.setObjective(v7, GRB.MAXIMIZE)

model.optimize()

print(f'{M.x} {K.x} {N.x}')
print(f'{m.x} {k.x} {n.x}')