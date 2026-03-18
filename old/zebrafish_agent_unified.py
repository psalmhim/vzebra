# ================================================================
# zebrafish_agent_unified.py
# ================================================================
# Unified Zebrafish Active Inference Agent (v1–v55)
# Integrating sensory, motor, dopaminergic, and metacognitive systems
# Author: H.-J. Park (2025)
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Import pre-defined modules
# ------------------------------------------------
from modules.retina_pc import RetinaPC
from modules.optic_tectum import OpticTectum
from modules.motor_tail import MotorTail
from modules.motor_eye import MotorEye
from modules.dopamine_system import DopamineSystem
from modules.meta_precision_field import MetaPrecisionField
from modules.temporal_inference_field import TemporalInferenceField
from modules.free_energy_engine import FreeEnergyEngine
from modules.thalamus_relay import ThalamusRelay
from modules.policy_field import PolicyField
from modules.visual_cortex_pc import VisualCortexPC
from modules.working_memory import WorkingMemory
from modules.deep_inference_field import DeepInferenceField
from modules.meta_memory import MetaMemory

# ================================================================
# Unified Active Inference Zebrafish Agent
# ================================================================

class ZebrafishAgent:
    def __init__(self, mode="full", seed=0):
        np.random.seed(seed)
        self.mode = mode

        # 1. Sensory modules
        self.retina = RetinaPC(mode="vision")
        self.tectum = OpticTectum(mode="predictive")

        # 2. Motor modules
        self.motor_tail = MotorTail(mode="bilateral")
        self.motor_eye = MotorEye(mode="tracking")

        # 3. Dopaminergic and valuation modules
        self.dopamine = DopamineSystem(mode="EFE")

        # 4. Higher inference modules
        self.temporal = TemporalInferenceField(mode="hierarchical")
        self.precision = MetaPrecisionField(mode="metacognitive")
        self.policy = PolicyField(mode="active")
        self.memory = WorkingMemory(mode="working")
        self.meta_memory = MetaMemory(mode="reflective")

        # 5. Core inference engine
        self.engine = FreeEnergyEngine(mode="full")

        # 6. Internal states
        self.F_hist = []
        self.precision_hist = []
        self.action_hist = []

        print(f"[Init] Zebrafish Agent created in mode = {mode}")

    # ------------------------------------------------------------
    # Perception–Action Loop
    # ------------------------------------------------------------
    def step(self, sensory_input):
        # Retina processing
        sensory_prediction = self.tectum.predict()
        sensory_error = self.retina.compute_error(sensory_input, sensory_prediction)

        # Temporal inference
        context = self.temporal.update_context(sensory_error)

        # Dopamine system computes expected free energy
        G = self.dopamine.compute_EFE(sensory_error, context)

        # Metacognitive precision control
        self.precision.update(G, sensory_error)

        # Policy selection and action
        action = self.policy.select_action(G, precision=self.precision.value)
        self.motor_tail.execute(action)

        # Memory consolidation
        self.memory.store(sensory_input, action, G)
        self.meta_memory.update_confidence(G)

        # Compute total free energy
        F = self.engine.compute_total_free_energy(sensory_error, G)
        self.F_hist.append(F)
        self.action_hist.append(action)
        self.precision_hist.append(self.precision.value)

        return F, action

    # ------------------------------------------------------------
    # Simulation run
    # ------------------------------------------------------------
    def run(self, env, T=500):
        for t in range(T):
            sensory_input = env.get_visual_input(t)
            F, action = self.step(sensory_input)
            env.update(action)
        return np.array(self.F_hist), np.array(self.action_hist), np.array(self.precision_hist)

# ================================================================
# Environments for version-based testing
# ================================================================

class VisualArena:
    def __init__(self, mode="static"):
        self.mode = mode
        self.target_pos = np.array([50.0, 0.0])
        self.fish_pos = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        print(f"[Env] Visual Arena mode = {mode}")

    def get_visual_input(self, t):
        # Mode-dependent stimulus pattern
        if self.mode == "v1-vision":
            return np.random.normal(0, 0.1)
        elif self.mode == "v10-pursuit":
            return np.linalg.norm(self.target_pos - self.fish_pos)
        elif self.mode == "v30-temporal":
            phase = np.sin(2 * np.pi * t / 50)
            return phase
        elif self.mode == "v50-metacognition":
            # occlusion or partial visual failure
            if 100 < t < 200:
                return np.nan
            else:
                return np.linalg.norm(self.target_pos - self.fish_pos)
        else:
            return 0.0

    def update(self, action):
        self.fish_pos += np.array([0.1 * action, 0])
        # simple motion update
        return self.fish_pos

# ================================================================
# Test Scenarios (v1–v55)
# ================================================================

def test_all_versions():
    tests = [
        ("v1-vision", 100),
        ("v10-pursuit", 200),
        ("v25-motor-learning", 300),
        ("v35-temporal-inference", 400),
        ("v45-confidence-control", 400),
        ("v55-metacognition", 500),
    ]

    for mode, T in tests:
        print(f"\n[TEST] Running mode = {mode}, steps = {T}")
        env = VisualArena(mode=mode)
        agent = ZebrafishAgent(mode="full")
        F, actions, precisions = agent.run(env, T=T)

        plt.figure(figsize=(8, 4))
        plt.subplot(3,1,1); plt.plot(F); plt.title(f"Free Energy over time ({mode})")
        plt.subplot(3,1,2); plt.plot(actions); plt.ylabel("Action")
        plt.subplot(3,1,3); plt.plot(precisions); plt.ylabel("Precision")
        plt.tight_layout()
        plt.show()

# ================================================================
# Run unified test suite
# ================================================================

if __name__ == "__main__":
    test_all_versions()
