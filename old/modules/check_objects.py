import matplotlib.pyplot as plt
import random, cv2, os

root = "synthetic_dataset"
fig, axs = plt.subplots(1, 5, figsize=(8, 2))
for i, cls in enumerate(["food", "prey", "plankton", "predator", "neutral"]):
    img = cv2.imread(os.path.join(root, cls, random.choice(os.listdir(os.path.join(root, cls)))), 0)
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(cls)
    axs[i].axis('off')
plt.tight_layout()
plt.show()
