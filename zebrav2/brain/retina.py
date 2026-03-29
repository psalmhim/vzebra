"""
4-type Retinal Ganglion Cell encoding.
Types: ON-sustained (RS), OFF-transient (CH), Looming (IB), Direction-selective (RS).
Input: intensity channel (400/eye), type channel (400/eye) from ray-caster.
Output: 4 spike-rate tensors per eye.
"""
import torch
import torch.nn as nn
import math
from zebrav2.spec import DEVICE, N_RET_PER_TYPE, N_RET_LOOM, N_RET_DS

class RetinaV2(nn.Module):
    """
    Encodes raw retinal input into 4 RGC type streams per eye.
    Uses rate-coded output (not full Izhikevich) for efficiency —
    true spiking happens in tectal layers downstream.
    """
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_on  = N_RET_PER_TYPE   # 150 per eye
        self.n_off = N_RET_PER_TYPE
        self.n_loom = N_RET_LOOM      # 50 per eye
        self.n_ds   = N_RET_DS        # 50 per eye
        # ON center-surround (Gaussian DoG filter approximation)
        # Store previous frame for OFF and motion computation
        self.register_buffer('prev_intensity_L', torch.zeros(400, device=device))
        self.register_buffer('prev_intensity_R', torch.zeros(400, device=device))
        # Looming: track apparent size history
        self.register_buffer('loom_size_L', torch.zeros(1, device=device))
        self.register_buffer('loom_size_R', torch.zeros(1, device=device))
        self.register_buffer('loom_vel_L', torch.zeros(1, device=device))
        self.register_buffer('loom_vel_R', torch.zeros(1, device=device))
        # Direction-selective: Reichardt delay buffer
        self.register_buffer('delay_L', torch.zeros(400, device=device))
        self.register_buffer('delay_R', torch.zeros(400, device=device))

    def encode_eye(self, intensity: torch.Tensor, prev_intensity: torch.Tensor,
                   delay: torch.Tensor, loom_size: torch.Tensor, loom_vel: torch.Tensor,
                   entity_type_pixels: dict) -> dict:
        """
        intensity: (400,) current frame intensity
        Returns dict with 'on', 'off', 'loom', 'ds' rates (each: target_n neurons)
        """
        # ON sustained: center excitation, sustained response
        on_raw = torch.clamp(intensity, 0, 1)
        # Simple pooling to n_on neurons
        on_rate = torch.nn.functional.adaptive_avg_pool1d(
            on_raw.unsqueeze(0).unsqueeze(0), self.n_on).squeeze()

        # OFF transient: respond to luminance decrease
        diff = prev_intensity - intensity  # positive when brightness drops
        off_raw = torch.clamp(diff * 3.0, 0, 1)  # amplify transients
        off_rate = torch.nn.functional.adaptive_avg_pool1d(
            off_raw.unsqueeze(0).unsqueeze(0), self.n_off).squeeze()

        # Looming: detect expanding entity (use enemy pixel count as proxy)
        enemy_px = entity_type_pixels.get('enemy', 0.0)
        new_size = torch.tensor([enemy_px / 50.0], device=self.device).clamp(0, 1)
        velocity = new_size - loom_size
        # l/v ratio: fire when size growing fast (velocity > 0) and size > threshold
        loom_trigger = torch.clamp(velocity * 5.0, 0, 1) * (new_size > 0.05).float()
        loom_rate = loom_trigger.expand(self.n_loom)

        # Direction selective: Reichardt detector (delayed correlation)
        ds_raw = intensity * delay  # correlation between current and delayed
        ds_raw = torch.clamp(ds_raw, 0, 1)
        ds_rate = torch.nn.functional.adaptive_avg_pool1d(
            ds_raw.unsqueeze(0).unsqueeze(0), self.n_ds).squeeze()

        return {
            'on': on_rate, 'off': off_rate,
            'loom': loom_rate, 'ds': ds_rate,
            'new_size': new_size, 'velocity': velocity
        }

    def forward(self, L: torch.Tensor, R: torch.Tensor,
                entity_info: dict = None) -> dict:
        """
        L, R: (800,) retinal input per eye (400 intensity + 400 type)
        entity_info: {'enemy_px': float, ...}
        Returns dict with L_on, L_off, L_loom, L_ds, R_on, R_off, R_loom, R_ds
        """
        if entity_info is None:
            entity_info = {}

        L_int, R_int = L[:400], R[:400]

        L_out = self.encode_eye(L_int, self.prev_intensity_L, self.delay_L,
                                 self.loom_size_L, self.loom_vel_L, entity_info)
        R_out = self.encode_eye(R_int, self.prev_intensity_R, self.delay_R,
                                 self.loom_size_R, self.loom_vel_R, entity_info)

        # Update history
        self.prev_intensity_L.copy_(L_int)
        self.prev_intensity_R.copy_(R_int)
        self.delay_L.copy_(L_int)
        self.delay_R.copy_(R_int)
        self.loom_size_L.copy_(L_out['new_size'])
        self.loom_size_R.copy_(R_out['new_size'])
        self.loom_vel_L.copy_(L_out['velocity'])
        self.loom_vel_R.copy_(R_out['velocity'])

        # Bilateral fused signals
        on_fused  = (L_out['on']  + R_out['on'])  / 2
        off_fused = (L_out['off'] + R_out['off']) / 2
        loom_fused = torch.max(L_out['loom'], R_out['loom'])
        ds_fused  = (L_out['ds']  + R_out['ds'])  / 2

        return {
            'L_on': L_out['on'], 'L_off': L_out['off'],
            'L_loom': L_out['loom'], 'L_ds': L_out['ds'],
            'R_on': R_out['on'], 'R_off': R_out['off'],
            'R_loom': R_out['loom'], 'R_ds': R_out['ds'],
            'on_fused': on_fused, 'off_fused': off_fused,
            'loom_fused': loom_fused, 'ds_fused': ds_fused,
            'loom_trigger': loom_fused.max().item() > 0.3,
        }

    def reset(self):
        self.prev_intensity_L.zero_()
        self.prev_intensity_R.zero_()
        self.delay_L.zero_()
        self.delay_R.zero_()
        self.loom_size_L.zero_()
        self.loom_size_R.zero_()
        self.loom_vel_L.zero_()
        self.loom_vel_R.zero_()
