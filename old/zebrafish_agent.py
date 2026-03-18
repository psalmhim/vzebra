# ============================================================
# MODULE: zebrafish_agent.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v55.5 (2025-12-07)
#
# PURPOSE:
#     Fully integrated zebrafish active inference agent:
#       Retina → RetinaPC → V1 → Memory → Dopamine
#       → TemporalInference → Policy → Motor
#
#     Supports:
#       - Supervised prey/predator classifier
#       - RL policy improvement
#       - Classifier-driven policy mode
# ============================================================

import torch
import torch.nn as nn

# Import retina renderer
from modules.retina_renderer import RetinaRenderer

# Vision modules
from modules.retina_pc import RetinaPC
from modules.visual_cortex_pc import VisualCortexPC

# Memory modules
from modules.working_memory import WorkingMemory
from modules.cortex_memory import CortexMemory

# Motor & midbrain modules
from modules.optic_tectum import OpticTectum
from modules.motor_tail import MotorTail
from modules.motor_eye import MotorEye
from modules.basal_ganglia import BasalGanglia

# Body & integration
from modules.body_pc import BodyPC
from modules.thalamus_relay import ThalamusRelay

# Active inference modules
from modules.temporal_inference_field import TemporalInferenceField
from modules.policy_field import GoalPolicyField
from modules.free_energy_engine import FreeEnergyEngine

# Dopamine systems
from modules.dopamine_tectal_system import DopamineTectalSystem
from modules.dopamine_hierarchical_efe_system import DopamineHierarchicalEFESystem


# ============================================================
# ZEBRAFISH AGENT
# ============================================================

class ZebrafishAgent:

    def __init__(
        self,
        device="cpu",
        n_pix=32,
        latent_dim=64,
        n_goals=3,
        use_classifier_policy=False
    ):
        self.device = device
        self.use_classifier_policy = use_classifier_policy

        # ---------------- Retina ----------------
        self.renderer = RetinaRenderer(device=device, n_features=n_pix)
        self.n_pix = n_pix
        self.n_features = 2 * n_pix       # ON+OFF

        self.retinaL = RetinaPC(n_in=self.n_features, latent_dim=latent_dim, device=device)
        self.retinaR = RetinaPC(n_in=self.n_features, latent_dim=latent_dim, device=device)

        # ---------------- Visual cortex ----------------
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)

        # ---------------- Memory ----------------
        self.mem = WorkingMemory(device=device)
        self.cmem = CortexMemory(device=device)

        # ---------------- Thalamus CMS ----------------
        self.thalamus = ThalamusRelay(device=device)

        # ---------------- Dopamine ----------------
        self.fastDA = DopamineTectalSystem(device=device)
        self.slowDA = DopamineHierarchicalEFESystem(device=device)

        # ---------------- Policy system ----------------
        self.temporal = TemporalInferenceField(
            mode="default",
            n_state=latent_dim,
            n_policy=n_goals,
            T=3,
            device=device
        )

        self.policy = GoalPolicyField(
            mode="active",
            n_goals=n_goals,
            device=device
        )

        # ---------------- Motor + tectum ----------------
        self.tectum = OpticTectum()
        self.mtail = MotorTail()
        self.meye = MotorEye()
        self.bg = BasalGanglia(mode="exploratory", device=device)

        # ---------------- Body ----------------
        self.body = BodyPC(device=device)

        # ---------------- Free-energy engine ----------------
        self.Fengine = FreeEnergyEngine(mode="sensory", device=device)

        # ---------------- Prey/predator classifier ----------------
        retinal_feature_dim = 4 * n_pix        # ON_L, OFF_L, ON_R, OFF_R
        self.prey_pred = nn.Linear(retinal_feature_dim, 2).to(device)

        self.last_action = None


    # ============================================================
    # STEP FUNCTION
    # ============================================================

    def step(self, retL, retR):

        # Retina predictive coding
        predL, FvL, _ = self.retinaL.step(retL)
        predR, FvR, _ = self.retinaR.step(retR)
        Fv = (FvL + FvR) / 2

        # Thalamus CMS
        cms = self.thalamus.step(Fv, Fv)

        # Fast dopamine
        da_fast, rpe_fast, surprise = self.fastDA.step(Fv, Fv)

        # Visual cortex encoding
        zL, _ = self.v1L.forward(predL)
        zR, _ = self.v1R.forward(predR)
        z = 0.5 * (zL + zR)

        # Memory update
        m, _ = self.mem.step(z, dopa=da_fast, cms=cms)

        # Build classifier input from raw retinal ON/OFF maps
        onL = retL["ON"].reshape(1, -1)
        offL = retL["OFF"].reshape(1, -1)
        onR = retR["ON"].reshape(1, -1)
        offR = retR["OFF"].reshape(1, -1)
        retinal_features = torch.cat([onL, offL, onR, offR], dim=1)

        # Classifier inference
        logits = self.prey_pred(retinal_features)
        prob = torch.softmax(logits, dim=1)
        prey_prob = prob[0, 0].item()
        pred_prob = prob[0, 1].item()

        # Slow hierarchical dopamine
        rpe_slow, da_slow, efe_slow, prec_slow = self.slowDA.step(Fv, Fv, 0.0, cms)

        # ======================================================
        # POLICY DECISION
        # ======================================================
        if self.use_classifier_policy:
            # Direct mapping from classifier → policy
            if prey_prob > pred_prob:
                choice = 0   # approach prey
            else:
                choice = 1   # flee predator
        else:
            # Full active inference policy
            motive, gain_state, choice, gvec, td_err, conf = self.policy.step(
                dopa=da_slow,
                rpe=rpe_slow,
                F_total=efe_slow,
                mem_mean=m.mean().item(),
                cms=cms
            )

        # ---------------- Motor commands ----------------
        if choice == 0:
            motor = self.mtail.approach_signal()
        elif choice == 1:
            motor = self.mtail.escape_signal()
        else:
            motor = self.mtail.neutral_signal()

        # Eye movement
        eye = self.tectum.step(prey_prob, pred_prob, Fv, motor, da_fast)

        return {
            "Fv": Fv,
            "cms": cms,
            "da_fast": da_fast,
            "da_slow": da_slow,
            "efe_slow": efe_slow,
            "prey_prob": prey_prob,
            "pred_prob": pred_prob,
            "eye": eye,
            "motor": motor,
            "policy": choice,
            "mem": m.mean().item()
        }


    # ============================================================
    # SUPERVISED CLASSIFIER TRAINING
    # ============================================================

    def train_classifier(self, X, y, epochs=15, lr=1e-3, device="cpu"):

        X = X.to(device)
        y = y.to(device).long()

        self.prey_pred.train()
        opt = torch.optim.Adam(self.prey_pred.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        print(f"\nTraining classifier: {len(X)} samples")
        print(f"  Prey: {(y==0).sum().item()}   Predator: {(y==1).sum().item()}")

        for ep in range(epochs):
            opt.zero_grad()
            logits = self.prey_pred(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            acc = (logits.argmax(1) == y).float().mean().item()

            if (ep+1) % 5 == 0 or ep == 0:
                print(f"[ep {ep+1:02d}] loss={loss.item():.4f}  acc={acc*100:.1f}%")

        print("✓ Classifier training complete\n")


    # ============================================================
    # RL UPDATE METHOD
    # ============================================================

    def rl_update_policy(self, prey_prob, pred_prob, choice, true_label, lr=0.05):
        """
        Reinforcement learning update:
          true_label: 0 = prey, 1 = predator
          choice: 0 = approach, 1 = flee, 2 = neutral
        Reward:
            +1 correct, -1 incorrect
        """
        correct = (true_label == 0 and choice == 0) or \
                  (true_label == 1 and choice == 1)

        reward = 1.0 if correct else -1.0

        # Update Q-values
        self.policy.Q[choice] += lr * reward
        self.policy.pref[choice] += 0.5 * lr * reward

        return reward
