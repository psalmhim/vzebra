"""
Conductance-based synaptic dynamics — sparse COO storage.

Weights stored as edge lists (row, col, val) instead of dense matrices.
Memory scales with nnz, not post_n × pre_n.
Forward uses index_add_ scatter: g_i = g_bar · Σ_j val_ij · s_j
"""
import torch
import torch.nn as nn
from zebrav2.spec import DT, DEVICE, TAU_AMPA, TAU_NMDA, TAU_GABA_A, E_AMPA, E_NMDA, E_GABA_A


class Synapse(nn.Module):
    """
    Single synapse type (AMPA, NMDA, or GABA_A) from pre_n → post_n neurons.
    Connectivity stored as sparse COO triplets (_row, _col, _val).
    """
    def __init__(self, pre_n: int, post_n: int, syn_type: str = 'AMPA',
                 g_bar: float = 1.0, device=DEVICE):
        super().__init__()
        self.pre_n = pre_n
        self.post_n = post_n
        self.syn_type = syn_type
        self.g_bar = g_bar
        self.device = device

        if syn_type == 'AMPA':
            self.tau = TAU_AMPA
            self.E_rev = E_AMPA
        elif syn_type == 'NMDA':
            self.tau = TAU_NMDA
            self.E_rev = E_NMDA
        elif syn_type == 'GABA_A':
            self.tau = TAU_GABA_A
            self.E_rev = E_GABA_A
        else:
            raise ValueError(f"Unknown synapse type: {syn_type}")

        self.decay = float(torch.exp(torch.tensor(-DT / self.tau)))

        # Sparse COO edge list: row=post index, col=pre index, val=weight
        self.register_buffer('_row', torch.empty(0, dtype=torch.long,    device=device))
        self.register_buffer('_col', torch.empty(0, dtype=torch.long,    device=device))
        self.register_buffer('_val', torch.empty(0, dtype=torch.float32, device=device))

        # Synaptic gating variable s per presynaptic neuron
        self.register_buffer('s', torch.zeros(pre_n, device=device))

    # ── COO builders ─────────────────────────────────────────────────────────

    def _from_dense(self, W: torch.Tensor):
        """Extract COO triplets from a dense weight matrix."""
        W = W.to(self.device)
        nz = W.nonzero(as_tuple=False)   # (nnz, 2)
        if len(nz) > 0:
            self._row = nz[:, 0]
            self._col = nz[:, 1]
            self._val = W[nz[:, 0], nz[:, 1]].clone()
        else:
            self._row = torch.empty(0, dtype=torch.long,    device=self.device)
            self._col = torch.empty(0, dtype=torch.long,    device=self.device)
            self._val = torch.empty(0, dtype=torch.float32, device=self.device)

    def init_sparse(self, p_connect: float, g_scale: float = 1.0):
        """Initialize sparse random connectivity (flat probability)."""
        mask = torch.rand(self.post_n, self.pre_n, device=self.device) < p_connect
        W    = torch.rand(self.post_n, self.pre_n, device=self.device) * g_scale
        self._from_dense(W * mask.float())

    def init_distance_sparse(self, pos_pre: torch.Tensor, pos_post: torch.Tensor,
                              lambda_um: float = 80.0, p_max: float = 0.15,
                              g_scale: float = 1.0):
        """Distance-dependent sparse connectivity: p_ij = p_max · exp(-d/λ).

        lambda_um=80 from PNAS empirical distance-correlation curve (Fig. S2B).
        """
        pos_pre  = pos_pre.to(self.device).float()
        pos_post = pos_post.to(self.device).float()
        a2   = (pos_post ** 2).sum(dim=1, keepdim=True)
        b2   = (pos_pre  ** 2).sum(dim=1, keepdim=True)
        dist = torch.sqrt(torch.clamp(a2 + b2.T - 2 * pos_post @ pos_pre.T, min=0.0))
        p_ij = (p_max * torch.exp(-dist / lambda_um)).clamp(0.0, 1.0)
        mask = torch.bernoulli(p_ij)
        W    = torch.rand(self.post_n, self.pre_n, device=self.device) * g_scale
        self._from_dense(W * mask)

    # ── Inspection helper ─────────────────────────────────────────────────────

    @property
    def W(self) -> torch.Tensor:
        """Return dense weight matrix (read-only view; do not use for in-place ops)."""
        w = torch.zeros(self.post_n, self.pre_n, device=self.device)
        if len(self._row) > 0:
            w[self._row, self._col] = self._val
        return w

    @property
    def nnz(self) -> int:
        return int(self._row.numel())

    def zero_weights(self):
        """Zero all synaptic weights (keeps topology, sets values to 0)."""
        self._val.zero_()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, pre_spikes: torch.Tensor, post_v: torch.Tensor) -> torch.Tensor:
        """
        pre_spikes: (pre_n,) binary
        post_v:     (post_n,) membrane potential
        Returns:    I_syn (post_n,) synaptic current in pA
        """
        self.s.copy_(self.s * self.decay + pre_spikes)

        # Sparse mat-vec via scatter: g_i = g_bar · Σ_j val_ij · s_j
        g = torch.zeros(self.post_n, device=self.device)
        if self.nnz > 0:
            g.index_add_(0, self._row, self._val * self.s[self._col])
        g = self.g_bar * g

        if self.syn_type == 'NMDA':
            mg_block = 1.0 / (1.0 + (1.0 / 3.57) * torch.exp(-0.062 * post_v))
            g = g * mg_block

        return g * (self.E_rev - post_v)

    def reset(self):
        self.s.zero_()
