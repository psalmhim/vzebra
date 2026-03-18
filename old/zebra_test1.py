from moving_entity_world import MovingEntityWorld
from zebrafish_agent import ZebrafishAgent
import matplotlib.pyplot as plt

world = MovingEntityWorld()
fish = ZebrafishAgent()

T=2000
prey_prob_curve=[]
pred_prob_curve=[]
beh_curve=[]
rew_curve=[]
dopa_curve=[]

for t in range(T):
    frame, label = world.step()
    out = fish.step(frame, label)

    prey_prob_curve.append(out["prey_p"])
    pred_prob_curve.append(out["pred_p"])
    beh_curve.append(out["behavior"])
    rew_curve.append(out["reward"])
    dopa_curve.append(out["dopa"])

plt.figure(figsize=(12,3)); plt.plot(prey_prob_curve,label="prey_p"); plt.plot(pred_prob_curve,label="pred_p"); plt.legend()
plt.figure(figsize=(12,3)); plt.plot(beh_curve); plt.title("0=Approach, 1=Flee, 2=Freeze")
plt.figure(figsize=(12,3)); plt.plot(rew_curve); plt.title("Reward")
plt.figure(figsize=(12,3)); plt.plot(dopa_curve); plt.title("Dopamine")
plt.show()

