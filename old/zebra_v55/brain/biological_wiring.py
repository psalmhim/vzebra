import torch

# =====================================================================
# Mask
# =====================================================================
def make_block_sparse_mask(n_in, n_out, blocks):
    mask = torch.zeros(n_in, n_out)
    in_bs  = n_in  // blocks
    out_bs = n_out // blocks
    for b in range(blocks):
        mask[b*in_bs:(b+1)*in_bs, b*out_bs:(b+1)*out_bs] = 1
    return mask


def apply_mask(layer, mask):
    with torch.no_grad():
        layer.W_ff *= mask.to(layer.W_ff.device)


# =====================================================================
# High-level retinotopic wiring modules
# =====================================================================
def apply_topographic_wiring(model):

    # Retina 400 -> OTL 600
    mask_L = make_block_sparse_mask(400, model.OTL, blocks=24)
    mask_R = make_block_sparse_mask(400, model.OTR, blocks=24)

    apply_mask(model.OT_L, mask_L)
    apply_mask(model.OT_R, mask_R)

    # OT_L+OT_R -> OT_fused
    mask_F = make_block_sparse_mask(model.OTL+model.OTR, model.OTF, blocks=32)
    apply_mask(model.OT_fused, mask_F)

    # OT_fused -> PT
    mask_PT = make_block_sparse_mask(model.OTF, model.PT, blocks=16)
    apply_mask(model.PT_layer, mask_PT)

    # PT -> PC_perc
    mask_PP = make_block_sparse_mask(model.PT, model.PC_PER, blocks=12)
    apply_mask(model.PC_perc, mask_PP)

    # PC_perc -> PC_intent
    mask_PI = make_block_sparse_mask(model.PC_PER, model.PC_INT, blocks=6)
    apply_mask(model.PC_intent, mask_PI)

    print("[SNN] Topographic block wiring applied.")


# =====================================================================
# Biological OT-L/R -> OT_fused mapping
# =====================================================================
def apply_biological_retina_to_OT(model):
    with torch.no_grad():
        W_L = model.OT_L.W_ff
        W_R = model.OT_R.W_ff
        W_L[:] = 0
        W_R[:] = 0
        for i in range(400):
            L_idx = (i * model.OTL) // 400
            R_idx = (i * model.OTR) // 400
            W_L[i, L_idx] = 1.0
            W_R[i, R_idx] = 1.0


def apply_biological_OT_fusion(model):
    OTL = model.OTL
    OTR = model.OTR
    OTF = model.OTF
    half = OTF // 2

    with torch.no_grad():
        W = model.OT_fused.W_ff
        W[:] = 0
        # Left block
        for i in range(OTL):
            out_idx = (i * half) // OTL
            W[i, out_idx] = 1.0
        # Right block
        offset = OTL
        for i in range(OTR):
            out_idx = half + (i * half) // OTR
            W[offset+i, out_idx] = 1.0

    print("[SNN] Biological OT_fused mapping applied.")


def apply_biological_PC_intent(model):
    PC_PER = model.PC_PER
    PC_INT = model.PC_INT
    half_int = PC_INT//2
    half_per = PC_PER//2

    with torch.no_grad():
        W = model.PC_intent.W_ff
        W[:] = 0
        # left perceptual → left intent
        for i in range(half_per):
            W[i, i % half_int] = 1.0
        # right perceptual → right intent
        for i in range(half_per, PC_PER):
            W[i, half_int + (i % half_int)] = 1.0

    print("[SNN] Biological PC mapping applied.")


def apply_biological_directionality(model):
    apply_biological_retina_to_OT(model)
    apply_biological_OT_fusion(model)
    apply_biological_PC_intent(model)
    print("[SNN] Direction-selective wiring applied.")


# =====================================================================
# Winner-take-all Motor decoding (robust)
# =====================================================================
def decode_brain_output(raw):
    """
    raw: dict with keys including 'intent', 'motor', 'CPG', 'eye', 'DA', 'per'
    """

    # ---------------------------------------
    # 1) PC_intent를 이용해 회전(turn) 계산
    # ---------------------------------------
    intent = raw["intent"][0]     # shape [PC_INT = 30]
    n = intent.numel() // 2       # 15

    left_int  = torch.relu(intent[:n]).mean()
    right_int = torch.relu(intent[n:]).mean()

    # normalized turn: -1 ~ +1
    turn = float((right_int - left_int) / (left_int + right_int + 1e-6))

    # ---------------------------------------
    # 2) forward drive from CPG
    # ---------------------------------------
    CPG = raw["CPG"][0]
    drive = float(torch.sigmoid(CPG).mean())

    # ---------------------------------------
    # 3) eye motor
    # ---------------------------------------
    e = torch.tanh(raw["eye"][0])
    eye_L = float(e[:50].mean())
    eye_R = float(e[50:].mean())

    # ---------------------------------------
    # 4) dopamine
    # ---------------------------------------
    DA = float(torch.sigmoid(raw["DA"][0]).mean())

    return {
        "turn_force": turn,
        "forward_drive": drive,
        "eye_L": eye_L,
        "eye_R": eye_R,
        "DA": DA,
        "tail_amp": drive
    }
