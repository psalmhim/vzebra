import numpy as np


# ======================================================================
# TailCPG: converts neural CPG spikes → traveling curvature wave
# ======================================================================

class TailCPG:
    """
    Tail CPG dynamics:
        - Receives CPG spike population from brain
        - Generates oscillatory signal
        - Propagates wave from head → tail
        - Produces scalar tail_amplitude + phase offset array
    """

    def __init__(self, n_joints=7):
        self.n_joints = n_joints

        # phase memory for each joint
        self.phase = np.zeros(n_joints, dtype=float)

        # base frequency
        self.base_freq = 14.0  # Hz typical zebrafish tail beat for cruising

        # internal oscillation state
        self.internal_phase = 0.0

    # ==================================================================
    # Update from CPG spikes
    # ==================================================================
    def update(self, spikes_CPG, DA_value, dt=0.02):
        """
        spikes_CPG: [1,200] torch tensor
        DA_value: vigor modulation (0~?)
        """
        # Convert spikes to equivalent firing rate
        rate = float(spikes_CPG.mean().item())

        # amplitude from rate + DA
        amp = max(0.1, min(2.0, 0.8 * rate + 0.6 * DA_value))

        # DA increases beat frequency slightly
        freq = self.base_freq * (1.0 + DA_value * 0.3)

        # integrate internal oscillation phase
        self.internal_phase += 2 * np.pi * freq * dt
        self.internal_phase %= (2 * np.pi)

        # traveling wave
        phases = np.zeros(self.n_joints, dtype=float)
        for i in range(self.n_joints):
            # posterior joints have extra phase delay
            phases[i] = self.internal_phase - (i * 0.35)

        return amp, phases
