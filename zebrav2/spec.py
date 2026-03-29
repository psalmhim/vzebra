"""Architecture constants — locked. Change requires commit justification."""
import torch

# Simulation
DT          = 0.001    # Izhikevich timestep (1 ms)
SUBSTEPS    = 50       # Izhikevich substeps per behavioral step (= 50ms per step)

import os

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        # Use CUDA_VISIBLE_DEVICES or pick last available GPU (least contention)
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if gpu_id is not None:
            return torch.device('cuda:0')  # maps to the visible device
        n_gpus = torch.cuda.device_count()
        return torch.device(f'cuda:{n_gpus - 1}')  # last GPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE      = _get_device()

# Synaptic time constants (seconds)
TAU_AMPA    = 0.005    # AMPA decay
TAU_NMDA    = 0.080    # NMDA decay
TAU_GABA_A  = 0.010    # GABA_A decay
TAU_GABA_B  = 0.080    # GABA_B decay

# Reversal potentials (mV)
E_AMPA      = 0.0
E_NMDA      = 0.0
E_GABA_A    = -80.0
E_GABA_B    = -90.0
E_REST      = -65.0

# Eligibility trace
TAU_ELIG    = 0.500    # eligibility trace decay (500 ms)

# Theta oscillation
T_THETA     = 0.125    # theta period (125 ms = 8 Hz)

# E/I connectivity probabilities
P_EE        = 0.10
P_EI        = 0.40
P_IE        = 0.40
P_II        = 0.10

# Conductances (nS)
G_EE        = 0.5
G_EI        = 2.0
G_IE        = 4.0
G_II        = 1.0

# Homeostatic
R_TARGET    = 0.05     # target mean spike rate per ms (~50Hz peak, sparse)
TAU_HOMEO   = 100000   # homeostatic timescale (steps)

# Tonic drive (pA)
I_TONIC_MU  = 3.0
I_TONIC_SIG = 1.0

# Layer sizes (~10K total, ~4× previous)
N_RET_PER_TYPE  = 400   # ON/OFF per eye (+ 100 loom + 100 DS = 1000/eye)
N_RET_LOOM      = 100
N_RET_DS        = 100
N_OT_SFGS_B     = 1200  # 900E + 300I (tectum is largest structure)
N_OT_SFGS_D     = 1200
N_OT_SGC        = 400   # 300E + 100I
N_OT_SO         = 400
N_PT            = 600   # 450E + 150I
N_TC            = 300
N_TRN           = 80
N_PAL_S         = 1600  # 1200E + 400I (pallium/cortex)
N_PAL_D         = 800   # 600E + 200I
N_D1            = 400
N_D2            = 300
N_GPI           = 60
N_RS            = 21    # reticulospinal (named, fixed)
N_CPG_PER_SIDE  = 48    # V2a+V0d+dI6+MN+Renshaw per side
N_PLACE         = 400   # 300E + 100I
N_DA            = 60
N_NA            = 30
N_5HT           = 40
N_ACH           = 30
