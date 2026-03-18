# eval_agent_behavior.py
import torch
from zebrafish_agent import ZebrafishAgent
from torchvision import transforms
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = ZebrafishAgent(device=device)
tfm = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

stimuli = ["food", "prey", "plankton", "predator", "neutral"]
results = []

for cls in stimuli:
    correct_behavior = {"food":0, "prey":0, "plankton":2, "predator":1, "neutral":2}[cls]
    correct = 0
    for i in range(1, 21):  # 20 test images
        img = tfm(Image.open(f"synthetic_dataset/{cls}/{i:04d}.png"))
        ret = {"ON": img.view(1, -1), "OFF": img.view(1, -1)}
        out = agent.step(ret, ret)
        if out["policy"] == correct_behavior:
            correct += 1
    acc = correct / 20
    results.append((cls, acc))

print("\nBehavioral accuracy:")
for r in results:
    print(f"{r[0]:>10}: {r[1]*100:.1f}%")

