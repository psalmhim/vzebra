# ============================================================
# MODULE: generate_synthetic_objects.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v3.1 (2025-12-07)
#
# PURPOSE:
#     Generate symbolic 2D images for zebrafish vision
#     with a clean fish-like predator silhouette.
# ============================================================

import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 28
N_PER_CLASS = 100
CLASSES = ["food", "prey", "plankton", "predator", "neutral"]


# ------------------------------------------------------------
# Fish-like predator: ellipse + tail triangle
# ------------------------------------------------------------
def draw_predator_fish(img, cx, cy):
    """Draw stylized fish-shaped predator (ellipse + triangle tail)."""
    body_len = np.random.randint(10, 12)
    body_wid = np.random.randint(6, 8)
    color = np.random.randint(200, 255)

    # Elliptical body
    cv2.ellipse(img, (cx, cy), (body_len, body_wid), 0, 0, 360, color, -1)

    # Tail triangle
    tail_len = 6
    pts = np.array([
        [cx - body_len, cy],
        [cx - body_len - tail_len, cy - 4],
        [cx - body_len - tail_len, cy + 4]
    ], np.int32)
    cv2.fillPoly(img, [pts], color)

    # Optional small eye for contrast (right side)
    if np.random.rand() > 0.5:
        cv2.circle(img, (cx + body_len // 2, cy - 2), 1, 0, -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    return img


# ------------------------------------------------------------
# Seaweed-like neutral background
# ------------------------------------------------------------
def draw_seaweed(img):
    """Draw wavy, branching vertical lines resembling seaweed."""
    for _ in range(np.random.randint(3, 6)):
        x = np.random.randint(4, IMG_SIZE - 4)
        y = IMG_SIZE - 1
        pts = []
        while y > 0:
            jitter = int(2 * np.sin(y / 3 + np.random.rand()))
            pts.append((x + jitter, y))
            y -= 1
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], np.random.randint(60, 120), 1)
    return img


# ------------------------------------------------------------
# Main shape drawer
# ------------------------------------------------------------
def draw_shape(img, cls):
    h, w = img.shape
    cx = np.random.randint(10, w - 10)
    cy = np.random.randint(10, h - 10)

    if cls == "food":
        for _ in range(np.random.randint(4, 8)):
            dx, dy = np.random.randint(-3, 3, 2)
            cv2.circle(img, (cx + dx, cy + dy), 1, 180, -1)

    elif cls == "prey":
        cv2.circle(img, (cx, cy), 3, 255, -1)
        
    elif cls == "plankton":
        cv2.circle(img, (cx, cy), 5, 150, -1)

    elif cls == "predator":
        img = draw_predator_fish(img, cx, cy)

    elif cls == "neutral":
        img = draw_seaweed(img)

    return img


# ------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------
def generate_dataset(base="synthetic_dataset"):
    base = Path(base)
    base.mkdir(exist_ok=True)

    for cls in CLASSES:
        out_dir = base / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[+] Generating {N_PER_CLASS} images for {cls}")

        for i in range(N_PER_CLASS):
            img = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
            img = draw_shape(img, cls)
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
            cv2.imwrite(str(out_dir / f"{i:04d}.png"), img)

        print(f"    ✓ Saved to {out_dir}")

    print("\n[✓] Synthetic dataset generation complete.")


if __name__ == "__main__":
    generate_dataset()
