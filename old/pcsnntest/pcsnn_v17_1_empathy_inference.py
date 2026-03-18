# ============================================================
# FILE: pcsnn_v17_1_empathy_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v17.1 (2025-11-17)
#
# PURPOSE:
#     Simulates two Active Inference agents with hierarchical
#     empathy inference and cooperative precision coupling.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules.retina_pc import RetinaPC
from modules.audio_pc import AudioPC
from modules.visual_cortex_pc import VisualCortexPC
from modules.working_memory import WorkingMemory
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_system import DopamineSystem
from modules.outcome_predictor import OutcomePredictor
from modules.policy_field import ActivePolicyField
from modules.social_field import SocialField
from modules.empathy_field import EmpathyField
from modules.basal_ganglia import BasalGanglia


class EmpathicAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(n_latent=64, n_goal=3, device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(device=device)
        self.outcome = OutcomePredictor(n_goals=3, device=device)
        self.policy = ActivePolicyField(n_goals=3, device=device)
        self.bg = BasalGanglia(mode="exploratory")
        self.empathy = EmpathyField(n_goals=3, device=device)

    def perceive_and_act(self, img, peer_state):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)
        rL, F_L, _ = self.retinaL.step(xL)
        rR, F_R, _ = self.retinaR.step(xR)
        a_pred, F_A, _ = self.audio.step(tone)
        zL, _ = self.v1L(rL, rR)
        zR, _ = self.v1R(rR, rL)
        z_mean = 0.5 * (zL + zR)
        Fv, Fa, Fb = (F_L + F_R)/2, F_A, 0.01 * abs(tone.mean().item())
        cms = self.thal.step(Fv, Fa, Fb)

        # internal free-energy and dopamine
        rpe, dopa, efe_self, prec = self.dopa.step(Fv, Fa, Fb, cms)
        reward_exp, unc_exp = self.outcome.predict()
        obs_reward = torch.tensor([1/(1+Fv+Fa+Fb)]*3, device=self.device)
        obs_unc = torch.tensor([Fv+Fa+Fb]*3, device=self.device)
        efe_vec, post, choice, gvec, conf = self.policy.step(
            reward_exp, unc_exp, obs_reward, obs_unc, cms
        )
        rpe_r, rpe_u = self.outcome.update(choice, obs_reward[choice], obs_unc[choice])

        # update empathy field based on observed peer
        peer_efe, peer_conf, peer_choice = peer_state
        emp_val, peer_goal_belief, peer_efe_est, peer_conf_est = self.empathy.step(
            peer_efe, peer_conf, peer_choice
        )

        # modulate dopamine precision via empathy
        dopa_mod = dopa + 0.3 * emp_val
        mem_state, _ = self.mem.step(z_mean, dopa_mod, cms, gvec, gain_state=conf)
        eye = self.bg.step(efe_vec[0], efe_vec[1], dopa_mod, rpe, cms)

        return efe_self, dopa_mod, conf, emp_val, peer_conf_est, peer_efe_est, cms, choice


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agentA = EmpathicAgent(device=device)
    agentB = EmpathicAgent(device=device)
    social = SocialField(n_agents=2, cooperative=True, device=device)

    efeA_hist, efeB_hist, empA_hist, empB_hist, prec_hist = [], [], [], [], []

    peerA_state = (0.5, 0.5, 1)
    peerB_state = (0.5, 0.5, 1)

    for t, (img, _) in enumerate(loader):
        efeA, dopaA, confA, empA, peerConfA, peerEFEA, cmsA, choiceA = agentA.perceive_and_act(img, peerB_state)
        efeB, dopaB, confB, empB, peerConfB, peerEFEB, cmsB, choiceB = agentB.perceive_and_act(img, peerA_state)

        peerA_state = (efeA, confA, choiceA)
        peerB_state = (efeB, confB, choiceB)

        coupled, valence, prec = social.step([efeA, efeB])
        efeA_hist.append(coupled[0].item()); efeB_hist.append(coupled[1].item())
        empA_hist.append(empA); empB_hist.append(empB); prec_hist.append(prec)

        if t % 50 == 0:
            print(f"Step {t:03d}: EFE_A={efeA:+.3f} EFE_B={efeB:+.3f} "
                  f"EmpA={empA:+.3f} EmpB={empB:+.3f} Prec={prec:+.3f}")
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(efeA_hist,label="EFE_A"); plt.plot(efeB_hist,label="EFE_B"); plt.legend()
    plt.subplot(3,1,2); plt.plot(empA_hist,label="Empathy_A"); plt.plot(empB_hist,label="Empathy_B"); plt.legend()
    plt.subplot(3,1,3); plt.plot(prec_hist,label="Social Precision Coupling"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v17_1_empathy_inference.png")
    print("[Saved] plots/v17_1_empathy_inference.png")

if __name__ == "__main__":
    main()
