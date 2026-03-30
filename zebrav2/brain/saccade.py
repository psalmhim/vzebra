"""
Spiking saccade module: gaze control for active vision.

Zebrafish larvae make convergent saccades during prey capture
(Bianco & Engert 2015) and scanning saccades during exploration.

Architecture:
  4 RS neurons: gaze_left, gaze_right, gaze_up, gaze_down
  2 IB neurons: saccade trigger (left burst, right burst)

Gaze offset modifies the effective fish heading for retinal projection,
so the fish can "look" in a direction different from its body heading.

Inputs: tectum salience map, goal (forage=track food, flee=track predator)
Output: gaze_offset (radians), applied to sensory_bridge heading
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingSaccade(nn.Module):
    def __init__(self, n_dir=4, n_trigger=2, device=DEVICE):
        super().__init__()
        self.device = device
        self.dir_pop = IzhikevichLayer(n_dir, 'RS', device)
        self.trigger_pop = IzhikevichLayer(n_trigger, 'IB', device)
        self.dir_pop.i_tonic.fill_(-2.0)
        self.trigger_pop.i_tonic.fill_(-3.0)

        self.register_buffer('dir_rate', torch.zeros(n_dir, device=device))
        self.register_buffer('trigger_rate', torch.zeros(n_trigger, device=device))

        self.gaze_offset = 0.0  # radians relative to body heading
        self.saccade_active = False
        self._gaze_ema = 0.0
        self.n_dir = n_dir
        self.n_trigger = n_trigger

    @torch.no_grad()
    def forward(self, food_bearing: float, enemy_bearing: float,
                current_goal: int, salience_L: float, salience_R: float) -> dict:
        """
        food_bearing: relative angle to nearest food (-pi to pi)
        enemy_bearing: relative angle to predator
        current_goal: 0=FORAGE, 1=FLEE, 2=EXPLORE, 3=SOCIAL
        salience_L/R: total retinal activity left/right eye
        """
        # Target gaze direction based on goal
        if current_goal == 0:  # FORAGE: look toward food
            target_gaze = food_bearing * 0.3  # partial shift, not full saccade
        elif current_goal == 1:  # FLEE: look toward predator (to track it)
            target_gaze = enemy_bearing * 0.2
        elif current_goal == 2:  # EXPLORE: scanning saccades
            target_gaze = 0.5 * math.sin(self._gaze_ema * 3.0)  # oscillating scan
        else:  # SOCIAL: look toward conspecific (most salient)
            target_gaze = 0.3 * (salience_R - salience_L) / (salience_L + salience_R + 1e-8)

        # Spiking dynamics
        I_dir = torch.zeros(self.n_dir, device=self.device)
        I_dir[0] = max(0, target_gaze) * 10.0   # gaze left
        I_dir[1] = max(0, -target_gaze) * 10.0  # gaze right
        I_dir[2] = abs(target_gaze) * 3.0        # vertical (minimal)
        I_dir[3] = abs(target_gaze) * 3.0

        # Saccade trigger: fires for large gaze shifts
        I_trigger = torch.zeros(self.n_trigger, device=self.device)
        gaze_error = abs(target_gaze - self._gaze_ema)
        if gaze_error > 0.3:  # large shift needed
            I_trigger[0 if target_gaze > 0 else 1] = gaze_error * 15.0

        for _ in range(10):
            self.dir_pop(I_dir + torch.randn(self.n_dir, device=self.device) * 0.3)
            self.trigger_pop(I_trigger + torch.randn(self.n_trigger, device=self.device) * 0.3)

        self.dir_rate.copy_(self.dir_pop.rate)
        self.trigger_rate.copy_(self.trigger_pop.rate)

        # Gaze output: smooth EMA toward target
        left_drive = float(self.dir_rate[0])
        right_drive = float(self.dir_rate[1])
        spiking_gaze = (left_drive - right_drive) * 0.5

        # Saccade: fast shift if trigger fires
        self.saccade_active = float(self.trigger_rate.max()) > 0.05
        if self.saccade_active:
            alpha = 0.5  # fast saccade
        else:
            alpha = 0.1  # smooth pursuit

        self._gaze_ema = (1 - alpha) * self._gaze_ema + alpha * target_gaze
        self.gaze_offset = max(-0.5, min(0.5, self._gaze_ema))  # ±30° max

        return {
            'gaze_offset': self.gaze_offset,
            'saccade_active': self.saccade_active,
            'target_gaze': target_gaze,
        }

    def reset(self):
        self.dir_pop.reset()
        self.trigger_pop.reset()
        self.dir_rate.zero_()
        self.trigger_rate.zero_()
        self.gaze_offset = 0.0
        self.saccade_active = False
        self._gaze_ema = 0.0
