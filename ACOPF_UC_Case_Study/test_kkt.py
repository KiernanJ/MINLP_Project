import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

pg = cp.Variable(1)
u = cp.Parameter(1, nonneg=True)

# Degenerate bound when u=0
constraints = [
    pg >= 0.5 * u,
    pg <= 1.5 * u,
    pg == 1.0 + cp.Variable(1) # forced to meet demand but slack allows violation?
]
# Let's make a simpler one
xi = cp.Variable(1, nonneg=True)
slack_con = [pg + xi >= 1.0] # demand is 1.0
obj = cp.Minimize(0.1 * pg + 100.0 * xi)
prob = cp.Problem(obj, [pg >= 0.5 * u, pg <= 1.5 * u] + slack_con)

layer = CvxpyLayer(prob, parameters=[u], variables=[pg, xi])

u_t = torch.tensor([0.0], requires_grad=True) # Degenerate u=0
pg_t, xi_t = layer(u_t)
loss = 100.0 * xi_t.sum()
loss.backward()
print("Grad at u=0.0:", u_t.grad)

u_t2 = torch.tensor([1e-3], requires_grad=True) # Slightly positive u
pg_t2, xi_t2 = layer(u_t2)
loss2 = 100.0 * xi_t2.sum()
loss2.backward()
print("Grad at u=1e-3:", u_t2.grad)
