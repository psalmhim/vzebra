"""
Spiking cerebellum: granule-Purkinje-DCN forward model.

Zebrafish cerebellum anatomy:
  Granule cells (GC): sparse combinatorial encoding, RS type
  Purkinje cells (PC): inhibitory output, learns via climbing fibers, IB type
  Deep cerebellar nuclei (DCN): output, disinhibited when PC pauses
  Climbing fibers (CF): error signal from pallium prediction error
  Mossy fibers (MF): sensory/motor context from tectum

Function: predicts sensory consequences of motor commands.
Prediction error (CF) drives PC synaptic plasticity (LTD at active parallel fibers).
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingCerebellum(nn.Module):
    def __init__(self, n_gc=200, n_pc=50, n_dcn=20, n_mossy=128, n_io=8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_gc = n_gc
        self.n_pc = n_pc
        self.n_dcn = n_dcn
        self.n_io = n_io

        # Neuron populations
        self.GC = IzhikevichLayer(n_gc, 'RS', device)
        self.PC = IzhikevichLayer(n_pc, 'IB', device)
        self.DCN = IzhikevichLayer(n_dcn, 'RS', device)

        # GC: high threshold (sparse coding — only ~10% active)
        self.GC.i_tonic.fill_(-4.0)
        # PC: moderate tonic (baseline inhibition of DCN)
        self.PC.i_tonic.fill_(2.0)
        # DCN: moderate tonic (fires when PC pauses)
        self.DCN.i_tonic.fill_(1.0)

        # Mossy fiber → GC (feedforward, sparse)
        self.W_mf_gc = nn.Linear(n_mossy, n_gc, bias=False)
        nn.init.sparse_(self.W_mf_gc.weight, sparsity=0.7, std=0.5)
        self.W_mf_gc.to(device)

        # Parallel fibers: GC → PC (learnable, LTD by climbing fiber)
        self.W_pf = nn.Parameter(
            torch.rand(n_pc, n_gc, device=device) * 0.3)

        # PC → DCN (inhibitory)
        self.register_buffer('W_pc_dcn',
            -torch.ones(n_dcn, n_pc, device=device) * 0.5)

        # Mossy fiber → DCN (excitatory collateral)
        self.W_mf_dcn = nn.Linear(n_mossy, n_dcn, bias=False)
        nn.init.xavier_uniform_(self.W_mf_dcn.weight, gain=0.3)
        self.W_mf_dcn.to(device)

        # IO → PC topographic climbing fiber projection (1:1 mapping via interpolation)
        # Each IO neuron drives its topographically corresponding PC population.
        # Gaussian neighbourhood kernel: PC_i gets strongest drive from IO_j nearest in rank.
        self.W_io_pc = nn.Linear(n_io, n_pc, bias=False)
        with torch.no_grad():
            i_idx = torch.arange(n_pc, dtype=torch.float32, device=device) / max(n_pc - 1, 1)
            j_idx = torch.arange(n_io, dtype=torch.float32, device=device) / max(n_io - 1, 1)
            sigma = 2.0 / n_pc  # narrow Gaussian → near-1:1 mapping
            topo = torch.exp(-((i_idx.unsqueeze(1) - j_idx.unsqueeze(0)) ** 2) / (2 * sigma ** 2))
            topo = topo / (topo.sum(dim=1, keepdim=True) + 1e-8)
            self.W_io_pc.weight.copy_(topo)
        self.W_io_pc.to(device)

        # State
        self.register_buffer('gc_rate', torch.zeros(n_gc, device=device))
        self.register_buffer('pc_rate', torch.zeros(n_pc, device=device))
        self.register_buffer('dcn_rate', torch.zeros(n_dcn, device=device))
        self.register_buffer('prediction_error', torch.tensor(0.0, device=device))

        # Eligibility trace for parallel fiber LTD
        self.register_buffer('pf_elig', torch.zeros(n_pc, n_gc, device=device))

    @torch.no_grad()
    def forward(self, mossy_input: torch.Tensor,
                climbing_fiber: float = 0.0,
                climbing_fiber_rate: torch.Tensor = None,
                DA: float = 0.5) -> dict:
        """
        mossy_input: (n_mossy,) sensory/motor context (tectum SFGS-b rates)
        climbing_fiber: scalar fallback error (used only if climbing_fiber_rate is None)
        climbing_fiber_rate: (n_io,) per-IO firing rate — projected 1:1 to PCs via W_io_pc
        DA: dopamine level for gating plasticity
        Returns: prediction error, DCN output rate
        """
        # Pool mossy input to expected size
        if mossy_input.shape[0] != self.W_mf_gc.in_features:
            # Manual average pooling (MPS doesn't support non-divisible adaptive pool)
            src = mossy_input.detach()
            tgt_size = self.W_mf_gc.in_features
            indices = torch.linspace(0, src.shape[0] - 1, tgt_size, device=self.device).long()
            mossy_input = src[indices]

        # Mossy fiber drive to GC
        I_mf_gc = self.W_mf_gc(mossy_input.unsqueeze(0)).squeeze(0).detach()
        I_mf_gc = I_mf_gc * (8.0 / (I_mf_gc.abs().mean() + 1e-8))

        # Mossy fiber collateral to DCN
        I_mf_dcn = self.W_mf_dcn(mossy_input.unsqueeze(0)).squeeze(0).detach()
        I_mf_dcn = I_mf_dcn * (3.0 / (I_mf_dcn.abs().mean() + 1e-8))

        gc_spikes = torch.zeros(self.n_gc, device=self.device)
        pc_spikes = torch.zeros(self.n_pc, device=self.device)
        dcn_spikes = torch.zeros(self.n_dcn, device=self.device)

        for _ in range(20):  # reduced substeps
            # GC: mossy fiber driven, sparse
            sp_gc = self.GC(I_mf_gc + torch.randn(self.n_gc, device=self.device) * 0.5)
            gc_spikes += sp_gc

            # PC: parallel fiber (GC→PC) + per-PC climbing fiber (topographic IO→PC)
            I_pf = self.W_pf @ self.GC.rate * 15.0
            if climbing_fiber_rate is not None:
                I_cf = self.W_io_pc(climbing_fiber_rate) * 5.0  # (n_pc,) per-PC
            else:
                I_cf = torch.full((self.n_pc,), climbing_fiber * 5.0, device=self.device)
            sp_pc = self.PC(I_pf + I_cf)
            pc_spikes += sp_pc

            # DCN: mossy collateral - PC inhibition
            I_pc_dcn = self.W_pc_dcn @ self.PC.rate * 10.0
            sp_dcn = self.DCN(I_mf_dcn + I_pc_dcn)
            dcn_spikes += sp_dcn

        # Update rates
        self.gc_rate.copy_(self.GC.rate)
        self.pc_rate.copy_(self.PC.rate)
        self.dcn_rate.copy_(self.DCN.rate)

        # Parallel fiber LTD: CF-driven depression at active GC→PC synapses
        # Biological rule: when CF fires AND GC was active, weaken PF synapse
        _cf_scalar = float(climbing_fiber_rate.mean()) if climbing_fiber_rate is not None else climbing_fiber
        if _cf_scalar > 0.1:
            gc_active = (gc_spikes > 0).float()
            pc_active = (pc_spikes > 0).float()
            # Update eligibility
            self.pf_elig = 0.9 * self.pf_elig + torch.outer(pc_active, gc_active)
            # LTD: reduce PF weights where both CF and GC were active
            ltd = -0.002 * _cf_scalar * DA * self.pf_elig
            self.W_pf.data.add_(ltd)
            self.W_pf.data.clamp_(0.01, 1.0)
        else:
            self.pf_elig *= 0.95  # decay eligibility

        # Prediction error = how much CF had to correct
        self.prediction_error.fill_(_cf_scalar)

        gc_frac = float((gc_spikes > 0).float().mean())
        return {
            'dcn_rate': self.dcn_rate,
            'gc_sparsity': gc_frac,
            'pc_rate_mean': float(self.pc_rate.mean()),
            'prediction_error': climbing_fiber,
        }

    def reset(self):
        self.GC.reset()
        self.PC.reset()
        self.DCN.reset()
        self.gc_rate.zero_()
        self.pc_rate.zero_()
        self.dcn_rate.zero_()
        self.prediction_error.zero_()
        self.pf_elig.zero_()
