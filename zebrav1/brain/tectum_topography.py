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
    """Apply topographic mapping for 2-channel retinal input.

    Each eye input is 800-dim: [400 intensity | 400 type_encoding].
    Both channels share the same topographic spatial structure.
    The OT weight matrix is 800×600, built as two stacked 400×600 maps.
    """
    with torch.no_grad():
        # Left retina → left shifted (400 pixels per channel)
        W_L_channel = build_gaussian_rf_map_offset(
            400, model.OTL,
            center_shift=-0.15,
            sigma=0.10
        )
        # Stack for both channels: [intensity_map; type_map]
        # Type channel gets slightly lower weight (0.7x) so intensity dominates
        W_L = torch.cat([W_L_channel, 0.7 * W_L_channel], dim=0)

        # Right retina → right shifted
        W_R_channel = build_gaussian_rf_map_offset(
            400, model.OTR,
            center_shift=+0.15,
            sigma=0.10
        )
        W_R = torch.cat([W_R_channel, 0.7 * W_R_channel], dim=0)

        model.OT_L.W[:] = W_L.to(model.device)
        model.OT_R.W[:] = W_R.to(model.device)

    print("[v1] Tectum topographic mapping applied.")

def build_gaussian_rf_map_offset(n_in, n_out, center_shift, sigma):
    W = torch.zeros(n_in, n_out)
    xs_in  = np.linspace(0, 1, n_in)
    xs_out = np.linspace(0, 1, n_out)

    for i, x_in in enumerate(xs_in):
        for j, x_out in enumerate(xs_out):
            shifted = x_out + center_shift
            W[i, j] = np.exp(-((x_in - shifted)**2)/(2*sigma*sigma))
    return W
