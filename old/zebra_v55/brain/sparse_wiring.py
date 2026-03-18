import torch
import numpy as np


# ================================================================
# Utility: create a random sparse binary mask
# ================================================================
def random_sparse_mask(in_dim, out_dim, sparsity=0.1, device="cpu"):
    """
    Returns a {0,1} mask with given sparsity.
    sparsity=0.1 → 10% of connections survive.
    """
    mask = (torch.rand(in_dim, out_dim, device=device) < sparsity).float()
    return mask


# ================================================================
# Utility: block sparse (topographic) mask
# ================================================================
def block_sparse_mask(in_dim, out_dim, blocks=4, density=0.5, device="cpu"):
    """
    Divide both dimensions into 'blocks' and connect each block-to-block
    region with density.
    Produces structured topographic connectivity similar to OT tectal map.

    Example:
        in_dim=1200, out_dim=2400, blocks=6
    """
    mask = torch.zeros(in_dim, out_dim, device=device)

    in_block = in_dim // blocks
    out_block = out_dim // blocks

    for bi in range(blocks):
        for bo in range(blocks):
            if torch.rand(1).item() < 0.7:  # about 70% block connectivity
                i_start = bi * in_block
                i_end = (bi + 1) * in_block
                o_start = bo * out_block
                o_end = (bo + 1) * out_block

                # random density inside block
                block_mask = (torch.rand(i_end - i_start,
                                         o_end - o_start,
                                         device=device) < density).float()

                mask[i_start:i_end, o_start:o_end] = block_mask

    return mask


# ================================================================
# Utility: distance-based sparse mask
# ================================================================
def gaussian_topographic_mask(in_dim, out_dim, sigma=0.2, device="cpu"):
    """
    Creates a topographic Gaussian-like connectivity map:
        W_ij = exp(- (i - j)^2 / (2*sigma^2) )

    Works best when in_dim ≈ out_dim.
    """
    xs = torch.linspace(0, 1, in_dim, device=device)
    ys = torch.linspace(0, 1, out_dim, device=device)

    xmat = xs.view(-1, 1)
    ymat = ys.view(1, -1)

    dist = (xmat - ymat) ** 2
    gaussian = torch.exp(-dist / (2 * sigma * sigma))

    # normalize into binary mask (0/1)
    thresh = torch.quantile(gaussian.flatten(), 0.7)
    mask = (gaussian > thresh).float()
    return mask


# ================================================================
# Apply mask to TwoCompLayer feedforward weights
# ================================================================
def apply_mask_to_layer(layer, mask):
    """
    layer: TwoCompLayer
    mask: torch.{0,1} matrix of shape [n_in, n_out]
    """
    with torch.no_grad():
        layer.W_ff *= mask.to(layer.W_ff.device)


# ================================================================
# Combined wiring profiles for v55.1 Zebrafish brain
# ================================================================
def generate_brain_wiring(dim, device="cpu"):
    """
    dim : BrainDims object containing:
            .retL
            .OT_L
            .OT_R
            .OT_fused
            .PT
            .PC_perc
            .PC_intent
            .motor
            .CPG
            .DA
            .eye

    Returns dict of masks for:
        retina→OT_L
        retina→OT_R
        OT_L+OT_R → OT_fused
        OT_fused → PT
        PT → PC_perc
        PC_perc → PC_intent
        PC_intent → motor
        motor → CPG
        PC_intent → DA
        PC_intent → eye
    """

    masks = {}

    # ------------------------------------------------------------
    # Retina to OT (local topography)
    # ------------------------------------------------------------
    masks["retL_to_OTL"] = block_sparse_mask(
        dim.retL, dim.OT_L, blocks=10, density=0.25, device=device
    )
    masks["retR_to_OTR"] = block_sparse_mask(
        dim.retR, dim.OT_R, blocks=10, density=0.25, device=device
    )

    # ------------------------------------------------------------
    # OT_L + OT_R → OT_fused (binocular)
    # ------------------------------------------------------------
    masks["OT_fuse"] = block_sparse_mask(
        dim.OT_L + dim.OT_R, dim.OT_fused, blocks=12, density=0.2, device=device
    )

    # ------------------------------------------------------------
    # OT_fused → PT (optic flow)
    # ------------------------------------------------------------
    masks["OT_to_PT"] = block_sparse_mask(
        dim.OT_fused, dim.PT, blocks=8, density=0.3, device=device
    )

    # ------------------------------------------------------------
    # PT → PC perception
    # ------------------------------------------------------------
    masks["PT_to_PCp"] = gaussian_topographic_mask(
        dim.PT, dim.PC_perc, sigma=0.2, device=device
    )

    # ------------------------------------------------------------
    # PC perception → PC intent
    # ------------------------------------------------------------
    masks["PCp_to_PCi"] = gaussian_topographic_mask(
        dim.PC_perc, dim.PC_intent, sigma=0.3, device=device
    )

    # ------------------------------------------------------------
    # PC intent → motor
    # ------------------------------------------------------------
    masks["PCi_to_motor"] = block_sparse_mask(
        dim.PC_intent, dim.motor, blocks=2, density=0.6, device=device
    )

    # ------------------------------------------------------------
    # motor → CPG
    # ------------------------------------------------------------
    masks["motor_to_CPG"] = block_sparse_mask(
        dim.motor, dim.CPG, blocks=4, density=0.3, device=device
    )

    # ------------------------------------------------------------
    # PC intent → DA
    # ------------------------------------------------------------
    masks["PCi_to_DA"] = random_sparse_mask(
        dim.PC_intent, dim.DA, sparsity=0.2, device=device
    )

    # ------------------------------------------------------------
    # PC intent → eye control
    # ------------------------------------------------------------
    masks["PCi_to_eye"] = random_sparse_mask(
        dim.PC_intent, dim.eye, sparsity=0.3, device=device
    )

    return masks


# ================================================================
# Helper to apply all masks to a brain instance
# ================================================================
def apply_wiring(brain, masks):
    apply_mask_to_layer(brain.retL_to_OTL, masks["retL_to_OTL"])
    apply_mask_to_layer(brain.retR_to_OTR, masks["retR_to_OTR"])
    apply_mask_to_layer(brain.OT_fuse, masks["OT_fuse"])
    apply_mask_to_layer(brain.PT, masks["OT_to_PT"])
    apply_mask_to_layer(brain.PC_perc, masks["PT_to_PCp"])
    apply_mask_to_layer(brain.PC_intent, masks["PCp_to_PCi"])
    apply_mask_to_layer(brain.motor, masks["PCi_to_motor"])
    apply_mask_to_layer(brain.CPG, masks["motor_to_CPG"])
    apply_mask_to_layer(brain.DA_mod, masks["PCi_to_DA"])
    apply_mask_to_layer(brain.eye_ctrl, masks["PCi_to_eye"])


