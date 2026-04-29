#%%
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.insert(0, ".")
from src.model_binary import BinaryPredictor

d = torch.load('data/supervised_data_case14_uctest.pt', weights_only=False)
X_sup = d['X_sup_t']
Y_sup = d['Y_u_sup_t']

model = BinaryPredictor(14, 5)
opt = optim.Adam(model.parameters(), lr=5e-3)
bce = nn.BCELoss()

for ep in range(1500):
    model.train()
    opt.zero_grad()
    _, u_prob, _, _, _ = model(X_sup)
    loss = bce(u_prob, Y_sup)
    loss.backward()
    opt.step()
    if (ep + 1) % 250 == 0:
        print(f"Epoch {ep+1} Loss: {loss.item():.4f}")

model.eval()
_, u_prob, _, _, _ = model(X_sup)
u_bin = torch.round(u_prob)
acc = (u_bin == Y_sup).float().mean()
print(f"Final Accuracy: {acc.item():.2%}")
print("Pred 0:", u_bin[0].tolist())
print("True 0:", Y_sup[0].tolist())
print("Pred 10:", u_bin[10].tolist())
print("True 10:", Y_sup[10].tolist())

# %%
