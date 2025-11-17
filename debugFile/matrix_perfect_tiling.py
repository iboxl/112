# this file is prepared for project 026
# Created by iboxl

from gurobipy import Model, GRB, quicksum
import gurobipy as gp
from gurobipy import *

m = Model("matrix_multiplication")
m.setParam('NonConvex', 2)
m.setParam('Threads', 320)
# m.setParam("FeasibilityTol", 1e-9)
m.setParam("IntFeasTol", 1e-9)
# m.setParam('LogFile', 'gurobi.log')
# m.setParam('MIPFocus', 1)
# m.setParam(GRB.Param.PoolSolutions, 100)  # 保存最多10个解
# m.setParam(GRB.Param.PoolSearchMode, 2)  # 找到所有解
# 决策变量
I = range(10)
M = 100
N = 100
bigM = max(M,N)+5

x = m.addVars(I, 2, lb=0, ub=N, vtype=GRB.INTEGER, name='x')
y = m.addVars(I, 2, lb=0, ub=M, vtype=GRB.INTEGER, name='y')
S = m.addVars(I, lb=1, ub=M*N, vtype=GRB.INTEGER)
tag = m.addVars(I, vtype=GRB.BINARY)
w = m.addVar(lb=1, ub=len(I), vtype=GRB.INTEGER)
for i in I:
    m.addConstr(x[i,0] <= x[i,1]-1)
    m.addConstr(y[i,0] <= y[i,1]-1)
    m.addConstr(S[i] == (x[i,1] - x[i,0] )*(y[i,1] - y[i,0]))
    m.addConstr(S[i] <= 2000)
for i in I[:-1]:
    m.addConstr(tag[i] >= tag[i+1])

m.addConstr(quicksum(tag[i] for i in I)<=w)
# m.update()
for i in I:
    for j in I:
        if j > i:
            z = m.addVars(2, vtype=GRB.BINARY) #overlap
            tmp = m.addVars(2, lb=-M*N, ub=M*N, vtype=GRB.CONTINUOUS)
            m.addConstr(tmp[0] == (x[i,0]-x[j,1])*(x[i,1]-x[j,0]))
            m.addConstr(tmp[1] == (y[i,0]-y[j,1])*(y[i,1]-y[j,0]))
            m.addGenConstrIndicator(z[0], True, tmp[0]<=-1)
            m.addGenConstrIndicator(z[0], False, tmp[0]>=0)
            m.addGenConstrIndicator(z[1], True, tmp[1]<=-1)
            m.addGenConstrIndicator(z[1], False, tmp[1]>=0)
            valid = m.addVar(vtype=GRB.BINARY)
            m.addGenConstrAnd(valid, [tag[i], tag[j]])
            m.addGenConstrIndicator(valid, True, z[0]+z[1]<=1)
            """   需要更高效更简明的建模方法          """
            # m.addConstr(tag[i] + tag[j] + z[0] + z[1] <= 3)

m.addConstr( quicksum( S[i]*tag[i] for i in I) == M * N )
m.update()

m.setObjective(w, GRB.MINIMIZE)
# m.setObjective(0)
# m.computeIIS()
# m.write("model.ilp")
m.optimize()
nSolutions = m.SolCount
# seen_solutions = set()
# unique_solutions = []
# for i in range(nSolutions):
#     m.setParam(GRB.Param.SolutionNumber, i)
#     print(f"Solution {i+1}")
#     for v in m.getVars():
#         print(f"{v.VarName}: {v.Xn}")
print(f"Number of solutions found: {nSolutions}")
if m.status == GRB.Status.OPTIMAL:
    print("Optimal solution found")
    # 打印 A 的最优解
    for i in range(nSolutions):
        m.setParam(GRB.Param.SolutionNumber, i)
        print(f'w is {w.x}')
        for i in range(int(w.x)):
            print(f"A[{i}] = ", end="")
            print(f" [{x[i, 0].x}, {y[i, 0].x}] ", end="")
            print(f" [{x[i, 1].x}, {y[i, 1].x}] ", end="")
            print("")
else:
    print("No optimal solution found")