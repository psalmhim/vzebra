import torch
import numpy as np

def build_gaussian_rf_map(n_in=400, n_out=600, sigma=8.0):
    W = torch.zeros(n_in, n_out)
    xs_in = np.linspace(0, 1, n_in)
    xs_out = np.linspace(0, 1, n_out)

    for i, x_in in enumerate(xs_in):
        for j, x_out in enumerate(xs_out):
            W[i, j] = np.exp(-((x_in - x_out)**2)/(2*sigma*sigma))
    return W

def apply_tectal_topography(model):
    with torch.no_grad():
        W_L = build_gaussian_rf_map(model.RET//2, model.OTL, sigma=0.08)
        W_R = build_gaussian_rf_map(model.RET//2, model.OTR, sigma=0.08)

        model.OT_L.W[:] = W_L.to(model.device)
        model.OT_R.W[:] = W_R.to(model.device)

    print("[v60] Tectum topographic mapping applied.")
