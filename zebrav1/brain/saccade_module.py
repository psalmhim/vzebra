import torch
import numpy as np

class SaccadeStabilizer:
    def __init__(self, ks=0.4):
        self.ks = ks

    def apply(self, motor_eye, world_slip, current_eye_vel):
        correction = self.ks * (world_slip - current_eye_vel)
        return motor_eye + correction
