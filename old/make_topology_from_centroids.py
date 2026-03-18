# make_topology_from_centroids.py
"""
Build zebrafish hemispheric topology matrix (A_final) using anatomical centroids.
Includes visualization & diagnostic utilities.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# ---------------------------------------------------------
# 1) Load and preprocess
# ---------------------------------------------------------
def load_centroids(path_centroids, n_regions=72):
    """
    Load a plain numeric CSV with shape [N x 3], representing centroids.
    Assumes no header line; columns are x,y,z.
    """
    df = pd.read_csv(path_centroids, header=None)
    coords = df.to_numpy(dtype=float)

    if coords.shape[1] != 3:
        raise ValueError(f"Centroid file must have 3 columns (x,y,z), found {coords.shape[1]}")
    if coords.shape[0] != n_regions:
        print(f"Warning: centroids have {coords.shape[0]} rows; expected {n_regions}")

    print(f"Loaded centroids: shape={coords.shape} from {path_centroids}")
    return coords


def build_bilateral_atlas(atlas_csv, save_csv=None):
    """
    Build bilateral region table (1–72) from unilateral MPIN atlas.
    Grey_level 1–36 → L_, 37–72 → R_
    """
    df = pd.read_csv(atlas_csv)
    if "Abbr" not in df.columns:
        raise ValueError("Expected column 'Abbr' in the atlas CSV")

    left = df.copy()
    right = df.copy()

    left["Hemisphere"] = "L"
    right["Hemisphere"] = "R"

    left["Abbr"] = "L_" + left["Abbr"].astype(str).str.strip()
    right["Abbr"] = "R_" + right["Abbr"].astype(str).str.strip()

    left["Grey_level"] = np.arange(1, len(df) + 1)
    right["Grey_level"] = np.arange(len(df) + 1, 2 * len(df) + 1)

    bilateral = pd.concat([left, right], ignore_index=True)
    bilateral = bilateral[["Grey_level", "Brain_region", "Abbr", "Hemisphere"]]

    if save_csv:
        bilateral.to_csv(save_csv, index=False)
        print(f"Saved bilateral atlas to {save_csv}")

    print(f"Bilateral atlas built with {len(bilateral)} regions (1–72)")
    return bilateral


# ---------------------------------------------------------
# 2) Build hemisphere mapping and distance matrix
# ---------------------------------------------------------
def detect_hemisphere(names):
    hemi = []
    for n in names:
        if n.startswith("L_"): hemi.append("L")
        elif n.startswith("R_"): hemi.append("R")
        else: hemi.append("C")
    return hemi


def distance_matrix(coords):
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------
# 3) Functional mask builders
# ---------------------------------------------------------
def functional_masks(names, chi=0.8):
    """Return dictionary of masks for different pathways."""
    N = len(names)
    M = {}
    M["base"] = np.ones((N, N))

    # Vision system
    retina = [i for i, n in enumerate(names) if "OB" in n or "OE" in n]
    tectum = [i for i, n in enumerate(names) if "TeO" in n]
    DI = [i for i, n in enumerate(names) if n.endswith("P") or "DI" in n]
    Cb = [i for i, n in enumerate(names) if "Cb" in n]
    MOS = [i for i, n in enumerate(names) if "MOS" in n]
    SPN = [i for i, n in enumerate(names) if "SP" in n]
    IPN = [i for i, n in enumerate(names) if "IPN" in n]
    Th = [i for i, n in enumerate(names) if "Th" in n]
    HYP = [i for i, n in enumerate(names) if "Hr" in n or "Hc" in n or "HYP" in n]

    # Initialize mask with zeros
    mask = np.zeros((N, N))

    # Retina→Tectum (optic crossing fraction χ)
    for i in retina:
        hemi_i = names[i][0]
        for j in tectum:
            hemi_j = names[j][0]
            if hemi_i != hemi_j: mask[j, i] = chi     # contralateral
            else: mask[j, i] = (1 - chi)              # ipsilateral

    # Tectum→DI/Cb→MOS→SP
    for i in tectum:
        for j in DI + Cb:
            mask[j, i] = 0.8
    for i in DI + Cb:
        for j in MOS:
            mask[j, i] = 0.8
    for i in MOS:
        for j in SPN:
            mask[j, i] = 1.0

    # Commissural links (cross-hemisphere)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if ni[0] != nj[0] and ni[2:] == nj[2:]:
                mask[j, i] = 0.2  # modest commissure

    # Tectum↔Tectum, DI↔DI interlinks
    for i in tectum:
        for j in tectum:
            if i != j and names[i][0] != names[j][0]:
                mask[j, i] = 0.1
    for i in DI:
        for j in DI:
            if i != j and names[i][0] != names[j][0]:
                mask[j, i] = 0.3

    M["function"] = mask
    return M


# ---------------------------------------------------------
# 4) Combine distance kernel + functional mask
# ---------------------------------------------------------
def build_A_final(names, coords, sigma=150.0, chi=0.8, wmax=1.0):
    D = distance_matrix(coords)
    A_dist = np.exp(-D**2 / (2 * sigma**2)) * wmax
    M = functional_masks(names, chi=chi)
    A_func = M["function"]
    A_final = A_dist * (A_func + 1e-3)  # functional scaling; avoids full zeros
    A_final = torch.tensor(A_final, dtype=torch.float32)
    return A_final, A_dist, A_func


# ---------------------------------------------------------
# 5) Visualization utilities
# ---------------------------------------------------------
def plot_centroids(names, coords, figsize=(6,6)):
    hemi = detect_hemisphere(names)
    colors = ["red" if h=="L" else "blue" if h=="R" else "gray" for h in hemi]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=colors, s=40)
    for i,n in enumerate(names):
        ax.text(coords[i,0], coords[i,1], coords[i,2], n, fontsize=6)
    ax.set_title("Region Centroids (L=red, R=blue)")
    plt.show()


def plot_connectivity_matrix(A, names, title="Connectivity Matrix"):
    A_np = A.detach().cpu().numpy() if isinstance(A, torch.Tensor) else A
    plt.figure(figsize=(8,7))
    sns.heatmap(A_np, cmap="viridis", square=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel("Presynaptic")
    plt.ylabel("Postsynaptic")
    plt.show()


def summarize_connectivity(A, names):
    A_np = A.detach().cpu().numpy() if isinstance(A, torch.Tensor) else A
    in_strength = A_np.sum(axis=1)
    out_strength = A_np.sum(axis=0)
    df = pd.DataFrame({
        "region": names,
        "in_strength": in_strength,
        "out_strength": out_strength
    })
    print(df.describe())
    return df





# ---------------------------------------------------------
# 6) Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    atlas_path = "atlas/MPIN-Atlas_brain_region_Combined_brain_regions_index.csv"
    centroid_path = "atlas/centroids.csv"

    bilateral = build_bilateral_atlas(atlas_path, save_csv="atlas_bilateral.csv")
    names = bilateral["Abbr"].tolist()
    coords = load_centroids(centroid_path, n_regions=len(names))

    A_final, A_dist, A_func = build_A_final(names, coords, sigma=180.0, chi=0.8)
    plot_centroids(names, coords)
    plot_connectivity_matrix(A_final, names)



