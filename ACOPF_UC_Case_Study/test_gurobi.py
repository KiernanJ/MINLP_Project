import gurobipy as gp
m = gp.Model()
x = m.addVar(lb=0, ub=1)
m.addConstr(x >= 2)
m.setObjective(5 * x)
m.optimize()
if m.status == gp.GRB.INFEASIBLE:
    print("Infeasible! Relaxing...")
    m.feasRelaxS(0, True, False, True)
    m.optimize()
    print("Relaxed Obj:", m.ObjVal)
