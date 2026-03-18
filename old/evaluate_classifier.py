# ============================================================
# evaluate_classifier.py
# PURPOSE: Test trained prey/predator classifier
# ============================================================

import torch
from dataset_builder import build_dataset
from zebrafish_agent import ZebrafishAgent
import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_classifier(model_path="models/prey_pred.pt", device="cpu", N_test=500):
    """
    Evaluate trained classifier on test set.
    
    Args:
        model_path: path to saved classifier weights
        device: torch device
        N_test: number of test samples
    """
    
    print("=" * 70)
    print("CLASSIFIER EVALUATION")
    print("=" * 70)
    print()
    
    # Build test dataset
    print(f"Building test dataset ({N_test} samples)...")
    X_test, y_test = build_dataset(N=N_test, device=device)
    
    # Load agent and classifier
    agent = ZebrafishAgent(device=device)
    
    if os.path.exists(model_path):
        agent.prey_pred.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded classifier from: {model_path}")
    else:
        print(f"⚠ Warning: Model file not found at {model_path}")
        print("  Evaluating with random initialization...")
    
    print()
    
    # Evaluate
    agent.prey_pred.eval()
    with torch.no_grad():
        logits = agent.prey_pred(X_test)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        
        # Overall accuracy
        acc = (pred == y_test).float().mean().item()
        
        # Per-class metrics
        prey_mask = (y_test == 0)
        pred_mask = (y_test == 1)
        
        prey_acc = (pred[prey_mask] == 0).float().mean().item() if prey_mask.sum() > 0 else 0.0
        pred_acc = (pred[pred_mask] == 1).float().mean().item() if pred_mask.sum() > 0 else 0.0
        
        # Confidence (mean probability of predicted class)
        pred_probs = probs[range(len(pred)), pred]
        mean_confidence = pred_probs.mean().item()
        
        # Confusion matrix
        tp = ((pred == 1) & (y_test == 1)).sum().item()  # true predator
        tn = ((pred == 0) & (y_test == 0)).sum().item()  # true prey
        fp = ((pred == 1) & (y_test == 0)).sum().item()  # false predator
        fn = ((pred == 0) & (y_test == 1)).sum().item()  # false prey
    
    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Overall accuracy:     {acc*100:.1f}%")
    print(f"Prey recognition:     {prey_acc*100:.1f}%")
    print(f"Predator recognition: {pred_acc*100:.1f}%")
    print(f"Mean confidence:      {mean_confidence:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Prey (0):      TP={tn:3d}  FN={fn:3d}")
    print(f"  True Predator (1):  FP={fp:3d}  TP={tp:3d}")
    print("=" * 70)
    print()
    
    # Plot confidence distribution
    os.makedirs("plots", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Confidence distribution
    ax = axes[0]
    prey_conf = probs[prey_mask, 0].cpu().numpy()
    pred_conf = probs[pred_mask, 1].cpu().numpy()
    
    ax.hist(prey_conf, bins=30, alpha=0.6, label='Prey', color='green')
    ax.hist(pred_conf, bins=30, alpha=0.6, label='Predator', color='red')
    ax.set_xlabel('Confidence (P(correct class))')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confusion matrix heatmap
    ax = axes[1]
    conf_mat = np.array([[tn, fn], [fp, tp]])
    im = ax.imshow(conf_mat, cmap='Blues', aspect='auto')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Prey', 'Pred: Predator'])
    ax.set_yticklabels(['True: Prey', 'True: Predator'])
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, conf_mat[i, j],
                          ha="center", va="center", color="black", fontsize=14)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("plots/classifier_evaluation.png", dpi=150)
    print("✓ Saved evaluation plot: plots/classifier_evaluation.png")
    print()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    evaluate_classifier(device=device, N_test=500)
