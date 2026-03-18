import torch

# =====================================================================
# Motor/CPG/Eye/Dopamine decoding
# =====================================================================

def decode_motor_outputs(brain_out):
    """
    Converts population spikes into continuous behavior parameters.

    brain_out contains:
        "motor": [1,200]
        "CPG": [1,200]
        "eye": [1,100]
        "DA": [1,50]

    Returns:
        dict with:
            turn_force      : float
            forward_drive   : float
            tail_amp        : float
            eye_L, eye_R    : floats
            DA              : float
    """

    motor = brain_out["motor"]   # [1,200]
    cpg   = brain_out["CPG"]     # [1,200]
    eye   = brain_out["eye"]     # [1,100]
    DA    = brain_out["DA"]      # [1,50]

    # ===============================================================
    # 1) TURNING FORCE
    # ===============================================================
    # Left-turn motor neurons = motor[0:100]
    # Right-turn motor neurons = motor[100:200]
    left_pop  = torch.sigmoid(motor[:, 0:100]).sum()
    right_pop = torch.sigmoid(motor[:, 100:200]).sum()
    turn_force = float((left_pop - right_pop) / 100.0)  # normalized difference

    # Scaling to -1 ~ +1
    turn_force = max(-1.0, min(1.0, turn_force * 2.0))


    # ===============================================================
    # 2) FORWARD DRIVE
    # ===============================================================
    # Based on overall motor firing
    forward_drive = float(motor.mean().item())

    # normalize
    forward_drive = max(0.0, min(forward_drive, 1.0))


    # ===============================================================
    # 3) TAIL AMPLITUDE
    # ===============================================================
    tail_amp = float(cpg.mean().item())
    tail_amp = max(0.0, min(tail_amp * 2.0, 2.0))   # scale 0~2


    # ===============================================================
    # 4) EYE ROTATION
    # ===============================================================
    eye_L = float(eye[:, 0:50].mean().item())
    eye_R = float(eye[:, 50:100].mean().item())
    eye_L = eye_L * 0.5
    eye_R = eye_R * 0.5


    # ===============================================================
    # 5) Dopamine modulation (vigor)
    # ===============================================================
    DA_val = float(DA.mean().item())
    DA_val = max(0.0, DA_val)


    return {
        "turn_force": turn_force,
        "forward_drive": forward_drive,
        "tail_amp": tail_amp,
        "eye_L": eye_L,
        "eye_R": eye_R,
        "DA": DA_val,
    }

