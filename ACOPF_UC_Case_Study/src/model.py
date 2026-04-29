"""
src/model.py
QCAC_Surrogate neural network — moved here from the notebook so that
ProcessPoolExecutor worker processes can import it cleanly.
"""

import torch
import torch.nn as nn


class QCAC_Surrogate(nn.Module):
    """
    Multi-head surrogate for the AC Unit Commitment problem.

    Predicts four quantities from a load profile (Pd, Qd):
      - u_pred   : continuous generator commitment probabilities  [N_g]
      - vr_base  : real-part voltage expansion points            [N_b]
      - vi_base  : imag-part voltage expansion points            [N_b]
      - rho      : penalty weight for the QCAC slack             [1]
      - A_cut    : learned integer-cut matrix                    [K × N_g]
      - b_cut    : learned integer-cut RHS                       [K]
    """

    def __init__(self, num_buses: int, num_gens: int, num_cuts: int = 5):
        super().__init__()
        self.num_buses = num_buses
        self.num_gens  = num_gens
        self.num_cuts  = num_cuts

        input_dim = num_buses * 2

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Head 1 — Voltage Taylor expansion points
        self.head_v    = nn.Linear(64, num_buses * 2)

        # Initialize voltage head: vr ≈ 1.0 (flat start), vi ≈ 0.0
        # This makes the Taylor expansion tight from epoch 1
        nn.init.zeros_(self.head_v.weight)
        with torch.no_grad():
            bias_v = torch.zeros(num_buses * 2)
            bias_v[:num_buses] = 1.0   # vr_base ≈ 1.0 per-unit
            # vi_base stays 0.0
            self.head_v.bias.copy_(bias_v)

        # Head 2 — Dynamic rho penalty (scalar, strictly positive)
        self.head_rho  = nn.Linear(64, 1)

        # Head 3 — Learned integer cuts  (A: K×N_g,  b: K)
        self.head_cuts = nn.Linear(64, (num_cuts * num_gens) + num_cuts)

        # Cuts start non-binding so they don't restrict any generators at init.
        # A_cut rows = all-ones, b_cut = num_gens  →  sum(u) <= num_gens + s_cut
        # which is always satisfied (u in [0,1]^N_g).  The network weight
        # (std=0.05) gives enough input-dependence so different load profiles
        # produce different cut tightenings from the first epoch.
        nn.init.normal_(self.head_cuts.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            bias_cuts = torch.zeros((num_cuts * num_gens) + num_cuts)
            bias_cuts[:num_cuts * num_gens] = 1.0          # A_cut = all-ones
            bias_cuts[num_cuts * num_gens:] = float(num_gens)  # b_cut = N_g (non-binding)
            self.head_cuts.bias.copy_(bias_cuts)
        
        # Initialize rho: must be > max(c0) so gens can't turn off to dodge no-load cost
        # c0 ranges from 200-1000, so rho=1000 ensures physical constraints > commitment cost
        nn.init.constant_(self.head_rho.bias, 5000.0)

    def forward(self, pd_qd_tensor):
        feat = self.trunk(pd_qd_tensor)

        v_out   = self.head_v(feat)
        vr_base = v_out[:, :self.num_buses]
        vi_base = v_out[:, self.num_buses:]

        # Clamp rho to [500, inf).
        # Break-even analysis: typical case14 generation cost ~ 1275 pu,
        # QCAC slack needed to absorb a u=0 power-balance mismatch ~ 2.5 pu,
        # so rho must exceed 1275/2.5 ~ 510 to make physical feasibility
        # more valuable than cost savings from turning generators off.
        # The sum(u)>=1 constraint in the CVXPY layer already prevents u=0,
        # but a high rho floor also keeps the Taylor linearisation tight.
        rho     = nn.functional.softplus(self.head_rho(feat))

        cuts_out = self.head_cuts(feat)
        A_cut    = cuts_out[:, :self.num_cuts * self.num_gens].view(-1, self.num_cuts, self.num_gens)
        b_cut    = cuts_out[:, self.num_cuts * self.num_gens:]

        return vr_base, vi_base, rho, A_cut, b_cut
