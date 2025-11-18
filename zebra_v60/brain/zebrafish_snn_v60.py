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

        self.RET = 800
        self.OTL = 600
        self.OTR = 600
        self.OTF = 800
        self.PT  = 400
        self.PC_PER = 120
        self.PC_INT = 30

        self.OT_L = TwoComp(400, self.OTL, device)
        self.OT_R = TwoComp(400, self.OTR, device)
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

        self.saccade = SaccadeStabilizer()

        apply_tectal_topography(self)

    def reset(self):
        self.OT_L.v.zero_()
        self.OT_R.v.zero_()
        self.OT_F.v.zero_()
        self.PT_L.v.zero_()
        self.PC_per.v.zero_()
        self.PC_int.v.zero_()

    @torch.no_grad()
    def forward(self, pos, heading, world):
        L, R = sample_retina_binocular_v60(pos, heading, world, device=self.device)
        # OT left/right
        oL = self.OT_L.step(L)
        oR = self.OT_R.step(R)
        # lateral inhibition
        oL, oR = lateral_inhibition_pair(oL, oR, lam=0.25)
        fused = torch.cat([oL, oR], dim=1)
        oF = self.OT_F.step(fused)

        # apply precision
        pi_OT = self.prec_OT.compute_pi()
        oF = pi_OT * oF

        pt = self.PT_L.step(oF)
        per = self.PC_per.step(pt)
        per = self.prec_PC.compute_pi() * per

        intent = self.PC_int.step(per)

        m = self.mot(intent)
        e = self.eye(intent)
        d = self.DA(intent)

        return {
            "motor": m,
            "eye": e,
            "DA": d
        }
