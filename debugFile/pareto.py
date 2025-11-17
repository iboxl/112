from gurobipy import Model, GRB

# 创建模型
m = Model("pareto_example")

# 创建变量
x = m.addVar(name="x")
y = m.addVar(name="y")
res = m.addVar(name="r")

# 添加约束
m.addConstr(x + y <= 10, "c0")

# 初始化解的列表
solutions = []

# 改变权重并求解
I = range(10)
for weight in I:  # 替换为你想要的权重列表
    m.setObjective((10 - weight) * x + weight * y, GRB.MAXIMIZE)
    m.optimize()
    if m.status == GRB.Status.OPTIMAL:
        solutions.append((m.getVarByName('x').X, m.getVarByName('y').X))

# 打印并分析解
for i, (x_val, y_val) in enumerate(solutions):
    print(f"Solution {i+1}: x = {x_val}, y = {y_val}")