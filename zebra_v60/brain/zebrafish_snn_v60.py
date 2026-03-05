import torch
import torch.nn as nn
from .retina_sampling_v60 import sample_retina_binocular_v60
from .tectum_topography_v60 import apply_tectal_topography
from .pc_precision_v60 import PrecisionUnit
from .lateral_inhibition_v60 import lateral_inhibition_pair
from .saccade_module_v60 import SaccadeStabilizer

class TwoComp(nn.Module):
    def __init__(self, n_in, n_out, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(n_in, n_out, device=device))
        self.v = torch.zeros(1, n_out, device=device)
        self.device = device
    def step(self, x):
        self.v = 0.8*self.v + 0.2*(x @ self.W)
        return self.v

class ZebrafishSNN_v60(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        # Retina: 800 per eye (400 intensity + 400 type channel)
        self.RET = 800
        self.OTL = 600
        self.OTR = 600
        self.OTF = 800
        self.PT  = 400
        self.PC_PER = 120
        self.PC_INT = 30

        # OT left/right receive full 800-dim retinal input
        self.OT_L = TwoComp(self.RET, self.OTL, device)
        self.OT_R = TwoComp(self.RET, self.OTR, device)
        self.OT_F = TwoComp(self.OTL+self.OTR, self.OTF, device)
        self.PT_L = TwoComp(self.OTF, self.PT, device)
        self.PC_per = TwoComp(self.PT, self.PC_PER, device)
        self.PC_int = TwoComp(self.PC_PER, self.PC_INT, device)

        # precision units
        self.prec_OT = PrecisionUnit(self.OTF, device)
        self.prec_PC = PrecisionUnit(self.PC_PER, device)

        # output heads
        self.mot = nn.Linear(self.PC_INT, 200)
        self.eye = nn.Linear(self.PC_INT, 100)
        self.DA  = nn.Linear(self.PC_INT, 50)

        # Classification head: branches from bilateral retinal type channels
        # Uses type-encoding channels (400 per eye = 800 total) which carry
        # entity-specific spectral signatures without foveation distortion
        self.cls_hidden = nn.Linear(800, 64)
        self.cls_out = nn.Linear(64, 5)

        self.saccade = SaccadeStabilizer()

        # Move all parameters to the target device
        self.to(device)

        apply_tectal_topography(self)

    def reset(self):
        """Reset all state tensors, detaching from any computation graph."""
        self.OT_L.v = torch.zeros(1, self.OTL, device=self.device)
        self.OT_R.v = torch.zeros(1, self.OTR, device=self.device)
        self.OT_F.v = torch.zeros(1, self.OTF, device=self.device)
        self.PT_L.v = torch.zeros(1, self.PT, device=self.device)
        self.PC_per.v = torch.zeros(1, self.PC_PER, device=self.device)
        self.PC_int.v = torch.zeros(1, self.PC_INT, device=self.device)

    def forward(self, pos, heading, world, depth_shading=False, depth_scale=80.0):
        L, R = sample_retina_binocular_v60(
            pos, heading, world, device=self.device,
            depth_shading=depth_shading, depth_scale=depth_scale)
        # OT left/right (now 800-dim input with type channel)
        oL = self.OT_L.step(L)
        oR = self.OT_R.step(R)
        # lateral inhibition
        oL, oR = lateral_inhibition_pair(oL, oR, lam=0.25)
        fused = torch.cat([oL, oR], dim=1)
        oF = self.OT_F.step(fused)

        # apply precision
        pi_OT = self.prec_OT.compute_pi()
        oF_weighted = pi_OT * oF

        pt = self.PT_L.step(oF_weighted)
        per = self.PC_per.step(pt)
        pi_PC = self.prec_PC.compute_pi()
        per = pi_PC * per

        intent = self.PC_int.step(per)

        m = self.mot(intent)
        e = self.eye(intent)
        d = self.DA(intent)

        # Classification from bilateral retinal type channels (no gradient to SNN)
        # Type channels: L[:, 400:] and R[:, 400:]
        type_features = torch.cat([L[:, 400:], R[:, 400:]], dim=1)  # [1, 800]
        cls_h = torch.relu(self.cls_hidden(type_features))
        cls = self.cls_out(cls_h)

        # Store intermediate activations for free energy computation
        self._last_oL = oL
        self._last_oR = oR
        self._last_oF = oF
        self._last_fused = fused
        self._last_pi_OT = pi_OT
        self._last_pi_PC = pi_PC

        # Retinal intensity channels only (first 400 pixels per eye)
        retL_intensity = L[:, :400]
        retR_intensity = R[:, :400]

        return {
            "motor": m,
            "eye": e,
            "DA": d,
            "cls": cls,           # classification logits [1, 5]
            "retL": retL_intensity,
            "retR": retR_intensity,
            "retL_full": L,       # full 2-channel [1, 800]
            "retR_full": R,       # full 2-channel [1, 800]
            "oL": oL,
            "oR": oR,
            "oF": oF,
            "fused": fused,
            "pt": pt,             # PT_L output [1, 400]
            "per": per,           # PC_per output [1, 120]
            "intent": intent,     # PC_int output [1, 30]
            "pi_OT": pi_OT.mean().item(),
            "pi_PC": pi_PC.mean().item(),
        }

    def get_saveable_state(self):
        """Return all learned weights for checkpoint persistence."""
        return self.state_dict()

    def load_saveable_state(self, state):
        """Restore learned weights from a checkpoint."""
        self.load_state_dict(state)

    def compute_free_energy(self):
        """Compute free energy as precision-weighted prediction error at OT level."""
        if not hasattr(self, '_last_oF'):
            return 0.0
        oF = self._last_oF
        fused = self._last_fused
        reconstructed = oF @ self.OT_F.W.t()
        error = fused - reconstructed
        pi_OT = self._last_pi_OT.mean()
        F = 0.5 * pi_OT * (error ** 2).mean()
        return F.item()

@torch.no_grad()
def enforce_motor_directionality(model):
    W = model.mot.weight  # shape [200, 30]
    W[:] = 0.0

    left_size = model.PC_INT // 2
    right_size = model.PC_INT - left_size

    for mi in range(100):
        for pi in range(left_size):
            W[mi, pi] = 1.0

    for mi in range(100, 200):
        for pi in range(left_size, model.PC_INT):
            W[mi, pi] = 1.0

    print("[v60] Motor directional mapping applied.")
