import numpy as np
import torch

from ..brain.zebrafish_snn import ZebrafishSNN
from ..brain.decode_motor import decode_motor_outputs

from ..body.fish_body import FishBody
from ..body.fish_physics import FishPhysics
from ..body.fish_sensors import FishSensors
from ..body.tail_cpg import TailCPG


# ======================================================================
# ZebrafishAgent: full integration of SNN + body + physics + sensors
# ======================================================================

class ZebrafishAgent:
    def __init__(self, world, device="cpu",
                 use_sparse_wiring=True,
                 plasticity=False,
                 dt=0.02):

        self.world = world
        self.device = device
        self.dt = dt

        # --------------------------------------------------------------
        # Brain (PC-SNN v55.1)
        # --------------------------------------------------------------
        self.brain = ZebrafishSNN(device=device,
                                     use_sparse_wiring=use_sparse_wiring)
        self.brain.to(device)
        self.brain.set_plasticity(plasticity)

        # --------------------------------------------------------------
        # Body + physics
        # --------------------------------------------------------------
        self.body = FishBody()
        self.physics = FishPhysics()

        # --------------------------------------------------------------
        # Sensors + reflex
        # --------------------------------------------------------------
        self.sensors = FishSensors(self.physics)

        # --------------------------------------------------------------
        # Tail CPG dynamics
        # --------------------------------------------------------------
        self.tail_cpg = TailCPG(n_joints=self.body.N)

        # --------------------------------------------------------------
        # Behavior state memory
        # --------------------------------------------------------------
        self.turn_force = 0.0
        self.forward_drive = 0.0
        self.tail_amp = 0.0
        self.eye_L = 0.0
        self.eye_R = 0.0
        self.DA = 0.0

        # --------------------------------------------------------------
        # Optional: set initial position random
        # --------------------------------------------------------------
        self.physics.x = np.random.uniform(world.xmin+40, world.xmax-40)
        self.physics.y = np.random.uniform(world.ymin+40, world.ymax-40)
        self.physics.heading = np.random.uniform(0, np.pi*2)

        # brain init
        self.reset_state()


    # ==================================================================
    # Reset agent state
    # ==================================================================
    def reset_state(self):
        self.brain.reset_state(1)
        self.turn_force = 0.0
        self.forward_drive = 0.0
        self.tail_amp = 0.0
        self.DA = 0.0

        # ==================================================================
    # Perform one full step:
    #   retina → brain → behavior → physics → body curvature
    # ==================================================================
    @torch.no_grad()
    def step(self, T_brain=1):
        """
        Executes one behavioral step of the agent.
        T_brain: number of internal PC-SNN cycles per step.
        """

        # --------------------------------------------------------------
        # 1. Retina Sampling → PC-SNN forward
        # --------------------------------------------------------------
        pos = np.array([self.physics.x, self.physics.y], dtype=float)
        heading = float(self.physics.heading)

        brain_out = self.brain.step_with_retina(
            position=pos,
            heading=heading,
            world=self.world,
            T=T_brain
        )

        # --------------------------------------------------------------
        # 2. Decode motor, eye, DA outputs
        # --------------------------------------------------------------
        decoded = decode_motor_outputs(brain_out)
        turn_force      = decoded["turn_force"]
        forward_drive   = decoded["forward_drive"]
        tail_amp        = decoded["tail_amp"]
        eye_L           = decoded["eye_L"]
        eye_R           = decoded["eye_R"]
        DA              = decoded["DA"]

        # --------------------------------------------------------------
        # 3. Reflex signals (from sensors)
        # --------------------------------------------------------------
        reflex_turn, reflex_forward = self.sensors.compute_reflex(self.world)

        # blend: reflexes override small, but brain dominates overall
        turn_force    = turn_force + 0.6 * reflex_turn
        forward_drive = forward_drive + 0.3 * reflex_forward

        # clamp
        forward_drive = max(0.0, min(forward_drive, 1.0))
        turn_force    = max(-1.0, min(turn_force, 1.0))

        # store
        self.turn_force    = turn_force
        self.forward_drive = forward_drive
        self.tail_amp      = tail_amp
        self.eye_L         = eye_L
        self.eye_R         = eye_R
        self.DA            = DA

        # --------------------------------------------------------------
        # 4. Tail CPG update → curvature wave
        # --------------------------------------------------------------
        amp, phases = self.tail_cpg.update(
            spikes_CPG=brain_out["CPG"],
            DA_value=DA,
            dt=self.dt
        )

        # head→tail curvature update
        # (body holds the final curvature state)
        for i in range(self.body.N):
            osc = amp * np.sin(phases[i])
            bias = -turn_force * (i / self.body.N)
            self.body.curv[i] = 0.8 * self.body.curv[i] + 0.2 * (osc + bias)

        # --------------------------------------------------------------
        # 5. Physics update
        # --------------------------------------------------------------
        self.physics.apply_motor(
            turn_force=turn_force,
            forward_drive=forward_drive,
            dt=self.dt
        )
        self.physics.step(dt=self.dt)

        # --------------------------------------------------------------
        # 6. Eat food if close
        # --------------------------------------------------------------
        eaten = self.world.try_eat(self.physics.x, self.physics.y)
        if eaten > 0:
            # dopamine reward only when food is eaten
            self.DA += eaten * 0.4

        # Return useful info
        return {
            "pos": (self.physics.x, self.physics.y),
            "heading": self.physics.heading,
            "turn": turn_force,
            "drive": forward_drive,
            "DA": self.DA
        }

        # ==================================================================
    # Hook to update renderer (called externally)
    # ==================================================================
    def draw_on(self, renderer, fish_index):
        """
        Delegates drawing to world_renderer.
        fish_index: index of this fish in renderer's fish list.
        """
        renderer.draw_fish(fish_index)


    # ==================================================================
    # Retrieve the state for monitoring/logging
    # ==================================================================
    def get_state(self):
        return {
            "x": float(self.physics.x),
            "y": float(self.physics.y),
            "heading": float(self.physics.heading),
            "turn_force": float(self.turn_force),
            "forward_drive": float(self.forward_drive),
            "tail_amp": float(self.tail_amp),
            "DA": float(self.DA)
        }


# ======================================================================
# Utility: create multiple zebrafish agents in a world
# ======================================================================

def create_agents(world, n=1, device="cpu",
                  use_sparse_wiring=True,
                  plasticity=False,
                  dt=0.02):
    agents = []
    for _ in range(n):
        ag = ZebrafishAgent(world,
                            device=device,
                            use_sparse_wiring=use_sparse_wiring,
                            plasticity=plasticity,
                            dt=dt)
        agents.append(ag)
    return agents
