import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, ".")
from src.model_binary import BinaryPredictor

d = torch.load('data/supervised_data_case14_uctest.pt', weights_only=False)
X_sup = d['X_sup_t']
Y_sup = d['Y_u_sup_t']

model = BinaryPredictor(14, 5)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
bce = nn.BCELoss()

print("Training Phase 1 on X_sup...")
for ep in range(1500):
    model.train()
    opt.zero_grad()
    _, u_prob, _, _, _ = model(X_sup)
    loss = bce(u_prob, Y_sup)
    loss.backward()
    opt.step()

model.eval()
_, u_prob_train, _, _, _ = model(X_sup)
acc_train = (torch.round(u_prob_train) == Y_sup).float().mean()
print(f"Train Accuracy (X_sup): {acc_train.item():.2%}")

test_data = torch.load('data/test_benchmarkcase14_uctest.pt', weights_only=False)
X_test = torch.stack([torch.tensor(inst['x_tensor'], dtype=torch.float32) for inst in test_data])
Y_test = torch.tensor([inst['u_true'] for inst in test_data], dtype=torch.float32)

_, u_prob_test, _, _, _ = model(X_test)
acc_test = (torch.round(u_prob_test) == Y_test).float().mean()
print(f"Test Accuracy (Benchmark instances): {acc_test.item():.2%}")
print("Prediction on Test 0:", torch.round(u_prob_test)[0].tolist())
print("True on Test 0:      ", Y_test[0].tolist())
