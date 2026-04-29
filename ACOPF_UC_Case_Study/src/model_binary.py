"""
src/model_binary.py
NN that directly predicts binary commitment u ∈ {0,1} using
Straight-Through Estimator (STE) for gradient flow.
Also predicts (vr_base, vi_base, rho) for the NLP diff layer.
"""
import torch
import torch.nn as nn


class StraightThroughRound(torch.autograd.Function):
    """Round in forward, identity gradient in backward (STE)."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryPredictor(nn.Module):
    """
    Predicts binary u from load profile, plus voltage/rho params for NLP layer.
    Outputs:
      u_binary [B, N_g]  — hard 0/1 (STE gradients)
      u_prob   [B, N_g]  — soft sigmoid probabilities (for BCE loss)
      vr_base  [B, N_b]
      vi_base  [B, N_b]
      rho      [B, 1]
    """
    def __init__(self, num_buses, num_gens):
        super().__init__()
        self.num_buses = num_buses
        self.num_gens = num_gens
        inp = num_buses * 2

        self.trunk = nn.Sequential(
            nn.Linear(inp, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
        )
        # Binary head
        self.head_u = nn.Linear(64, num_gens)
        # Voltage head
        self.head_v = nn.Linear(64, num_buses * 2)
        nn.init.zeros_(self.head_v.weight)
        with torch.no_grad():
            b = torch.zeros(num_buses * 2)
            b[:num_buses] = 1.0
            self.head_v.bias.copy_(b)
        # Rho head
        self.head_rho = nn.Linear(64, 1)
        nn.init.constant_(self.head_rho.bias, 5000.0)

    def forward(self, x):
        feat = self.trunk(x)
        u_logit = self.head_u(feat)
        u_prob = torch.sigmoid(u_logit)
        u_binary = StraightThroughRound.apply(u_prob)
        v = self.head_v(feat)
        vr = v[:, :self.num_buses]
        vi = v[:, self.num_buses:]
        rho = nn.functional.softplus(self.head_rho(feat))
        return u_binary, u_prob, vr, vi, rho
