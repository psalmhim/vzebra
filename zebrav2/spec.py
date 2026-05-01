"""
Architecture constants — connectome-grounded (6-7 dpf larval zebrafish).

Biological references:
  Kunst et al. 2019  (Neuron 103) — cellular-resolution atlas, mapZebrain wiring
  Hildebrand et al. 2017 (Nature 545) — whole-brain serial-section EM
  2025 CLEM connectome (PMC12259123) — 180K soma, 39M synapses, 7 dpf
  Kölsch et al. 2021 (Neuron 109) — RGC subtypes (~4,000/eye, 29 types)
  Robles et al. 2011 (Front. Neural Circuits) — tectum E/I: 30-35% E, 65-70% I
  Zabegalov et al. 2022 (Front. Mol. Neurosci.) — 13,320 tectal cells, 25 types
  Bhatt et al. 2020 (J. Neurosci.) — PC ~300-400, EN ~190/hemi, I/E ratio 3.8:1
  Wullimann & Bhatt 2023 (Cell. Mol. Life Sci.) — PC plateau ~400 at 6 dpf
  Knogler et al. 2017 (Curr. Biol.) — ~13K active granule cells
  Hsieh et al. 2024 (J. Neurosci.) — ~100-150 IO neurons/hemisphere
  Pandey et al. 2018 (Curr. Biol.) — ~1,500 habenular neurons, 18 types
  Cosacak et al. 2023 (Genome Res.) — pallium/subpallium cell types at 6 dpf
  Gaspar & Bhatt 2014 — ~244 raphe cells; ~81 DRN 5-HT neurons

Real scale: ~100,000 neurons total (larval zebrafish, 6-7 dpf).
Computational scale here: ~34,000 neurons (~34% of real).
Tectum is largest structure (35-40% of real brain); correctly dominates here.
Cerebellum PC (400) and EN (380) and IO (300) are exact biological counts.
"""
import torch
import os


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if gpu_id is not None:
            return torch.device('cuda:0')
        n_gpus = torch.cuda.device_count()
        return torch.device(f'cuda:{n_gpus - 1}')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _get_device()

# ── Simulation ────────────────────────────────────────────────────────────────
DT       = 0.001   # Izhikevich timestep (1 ms)
SUBSTEPS = 50      # substeps per behavioral step (= 50 ms/step)

# ── Synaptic time constants (seconds) ────────────────────────────────────────
TAU_AMPA   = 0.005   # AMPA decay
TAU_NMDA   = 0.080   # NMDA decay
TAU_GABA_A = 0.010   # GABA_A fast decay
TAU_GABA_B = 0.080   # GABA_B slow decay

# ── Reversal potentials (mV) ──────────────────────────────────────────────────
E_AMPA   =   0.0
E_NMDA   =   0.0
E_GABA_A = -80.0
E_GABA_B = -90.0
E_REST   = -65.0

# ── Eligibility & oscillation ─────────────────────────────────────────────────
TAU_ELIG = 0.500   # eligibility trace decay (500 ms)
T_THETA  = 0.125   # theta period (125 ms = 8 Hz)

# ── Global E/I connectivity probabilities (local intra-region) ────────────────
# Robles 2011; realistic local cortical: P_EE ~0.05-0.10
P_EE = 0.08
P_EI = 0.20
P_IE = 0.20
P_II = 0.05

# ── Global synaptic conductances (nS) ─────────────────────────────────────────
G_EE = 0.5
G_EI = 2.0
G_IE = 4.0
G_II = 1.0

# ── Homeostatic ───────────────────────────────────────────────────────────────
R_TARGET   = 0.05    # target mean spike rate per ms (~50 Hz peak, sparse)
TAU_HOMEO  = 100000  # homeostatic timescale (steps)

# ── Tonic drive (pA) ──────────────────────────────────────────────────────────
I_TONIC_MU  = 3.0
I_TONIC_SIG = 1.0

# ── Per-region E/I fractions (E_fraction, I_fraction) ────────────────────────
# Used to override the global 75%E default for individual EILayer instances.
# Sources: Robles 2011 (tectum), Bhatt 2012 (thalamus), Pandey 2018 (habenula),
#          circuit anatomy (cerebellum, striatum).
EI_FRAC_DEFAULT = (0.75, 0.25)   # generic cortical (baseline)
EI_FRAC_OT      = (0.32, 0.68)   # optic tectum: 32% E, 68% I (Robles 2011)
EI_FRAC_TC      = (0.90, 0.10)   # thalamus: predominantly glutamatergic relay
EI_FRAC_HB_D    = (0.82, 0.18)   # dorsal habenula: 82% E (Pandey 2018)
EI_FRAC_HB_V    = (0.50, 0.50)   # ventral habenula: mixed
EI_FRAC_STR     = (0.00, 1.00)   # striatum D1/D2: 100% GABAergic MSNs
EI_FRAC_CB_PC   = (0.00, 1.00)   # Purkinje cells: 100% GABAergic
EI_FRAC_CB_GC   = (1.00, 0.00)   # granule cells: 100% glutamatergic
EI_FRAC_IO      = (1.00, 0.00)   # inferior olive: 100% excitatory climbing fibers

# ─────────────────────────────────────────────────────────────────────────────
# NEURON COUNTS
# Real biological values annotated. Computational values are ~34% of real
# (MPS feasibility). Exact real counts for well-characterised small populations.
# ─────────────────────────────────────────────────────────────────────────────

# ── RETINA ────────────────────────────────────────────────────────────────────
# Real: ~4,000 RGCs/eye (Kölsch 2021), 29 transcriptional subtypes.
# ~97% project to tectum; others to AF1-AF9 (pretectum, thalamus).
# ON/OFF ganglion cells share N_RET_PER_TYPE; loom & DS are separate types.
N_RET_PER_TYPE = 1000   # ON or OFF RGCs per eye  (real: ~1,600-2,000/subtype)
N_RET_LOOM     = 300    # loom/threat-detector RGCs per eye  (real: ~400)
N_RET_DS       = 400    # direction-selective RGCs per eye   (real: ~600)
# Both eyes total: (1000+1000+300+400)×2 = 5,400

# ── OPTIC TECTUM ──────────────────────────────────────────────────────────────
# Real: ~30,000-40,000 neurons bilateral (Zabegalov 2022: 13,320 sampled).
# E/I: 30-35% E, 65-70% I (Robles 2011 dlx5/6+ population).
# Layers (N_ = total bilateral; each hemisphere gets N//2):
N_OT_SFGS_B = 3600   # SFGS broad sublayers  (real bilateral: ~17,500)
N_OT_SFGS_D = 3000   # SFGS deep sublayers   (real bilateral: ~12,500)
N_OT_SGC    = 1200   # stratum griseum centrale — PVPNs + PVINs (real: ~3,750)
N_OT_SO     = 600    # stratum opticum — retinorecipient neuropil (real: ~1,750)
# Tectum total: 8,400

# ── PRETECTUM ─────────────────────────────────────────────────────────────────
# Real: ~500-1,500; receives ~50% of RGC collaterals (AF5-AF9).
N_PT = 1000   # (real: ~1,000-1,500)

# ── THALAMUS ──────────────────────────────────────────────────────────────────
# Real: ~1,000-2,000; predominantly glutamatergic relay (Bhatt 2012).
# Dorsal thalamus → tectum (10/37 non-retinal inputs; Kunst 2019).
N_TC  = 1000   # dorsal thalamus relay neurons  (real: ~1,000-2,000)
N_TRN = 300    # thalamic reticular nucleus, inhibitory  (real: ~300-500)

# ── CEREBELLUM ────────────────────────────────────────────────────────────────
# PC and EN counts are exact biological values (Bhatt 2020, Wullimann 2023).
# GC count is ~31% of real (real: ~13,000 functionally active, Knogler 2017).
N_CB_GC = 4000   # granule cells (real: ~13,000; limited by mossy fiber matrix)
N_CB_PC = 400    # Purkinje cells — EXACT (plateau ~400 at 6 dpf, Wullimann 2023)
N_CB_EN = 380    # eurydendroid cells — EXACT (~190/hemi bilateral, Bhatt 2020)
N_CB_GO = 150    # Golgi cells — inhibitory feedback onto GCs (real: ~200)
# Cerebellum total: 4,930

# ── INFERIOR OLIVE ────────────────────────────────────────────────────────────
# EXACT: ~100-150/hemisphere → ~300 bilateral (Hsieh 2024, J. Neurosci.).
# All projections contralateral; 1 climbing fiber per Purkinje cell.
N_IO = 300    # inferior olive bilateral — EXACT

# ── HABENULA ──────────────────────────────────────────────────────────────────
# EXACT total: ~1,500 neurons, 18 cell types (Pandey 2018, Curr. Biol.).
# dHb: left-right asymmetric; projects via fasciculus retroflexus → IPN.
# vHb: lateral habenula homolog; receives entopeduncular + hypothalamic input.
N_HB_D = 800    # dorsal habenula bilateral  (real: ~1,000)
N_HB_V = 400    # ventral habenula bilateral (real: ~500)
# Habenula total: 1,200 (real: ~1,500)

# ── PALLIUM / TELENCEPHALON ───────────────────────────────────────────────────
# Real: ~2,000-5,000 at 6 dpf (Cosacak 2023); 9 pallial types by scRNA-seq.
N_PAL_S = 2000   # superficial pallium (75% E; ~1,500 E + 500 I)
N_PAL_D = 1200   # deep pallium
N_PLACE  = 600   # place/spatial cells (hippocampal-like, within pallium)

# ── SUBPALLIUM / BASAL GANGLIA ────────────────────────────────────────────────
# D1/D2 MSNs: ~100% GABAergic (EI_FRAC_STR). No VTA/SNc in zebrafish.
# Dopamine from posterior tuberculum (DC2 group), not from midbrain.
N_D1  = 800    # D1 striatum, direct pathway   (real: ~1,200 estimate)
N_D2  = 600    # D2 striatum, indirect pathway (real: ~900 estimate)
N_GPI = 150    # GPi / entopeduncular nucleus  (real: ~200 estimate)

# ── OLFACTORY BULB ────────────────────────────────────────────────────────────
# Mitral + tufted cells (output neurons); receives from olfactory epithelium.
N_OB = 600    # (real: ~800-1,200)

# ── HYPOTHALAMUS ─────────────────────────────────────────────────────────────
# Real: ~2,000-4,000 combined. Rostral caudal = 3× more catecholaminergic.
# Contains oxytocinergic, CRH, and th2+ dopaminergic subpopulations.
N_HYP_R = 600    # rostral hypothalamus (Hr)
N_HYP_I = 600    # intermediate hypothalamus (Hi)
N_HYP_C = 800    # caudal hypothalamus (Hc) — most catecholaminergic
# Hypothalamus total: 2,000

# ── RAPHE NUCLEUS ─────────────────────────────────────────────────────────────
# Real: ~244 total raphe cells (functional imaging); ~81 DRN 5-HT neurons.
# Ascending SRa projects to forebrain; descending IRa reaches spinal cord.
N_RA = 200    # raphe total bilateral (real: ~244)

# ── MEDIAL OCTAVOLATERAL NUCLEUS (lateral line) ───────────────────────────────
# Real: ~750/hemisphere ≈ 1,500 bilateral (2025 CLEM connectome PMC12259123).
# ~43,000 afferent→MON synapses from ~150 PLLg afferents.
N_MON = 1000    # MON bilateral (real: ~1,500)

# ── TORUS SEMICIRCULARIS (auditory/LL midbrain) ───────────────────────────────
# Homolog of inferior colliculus; receives VIII-nerve + LL ascending input.
N_TS = 500    # (real: ~500-2,000; larval stage incompletely myelinated)

# ── RETICULAR FORMATION ───────────────────────────────────────────────────────
# Hildebrand 2017: 22 bilateral RS pairs (44 identified myelinated RS neurons).
# Kimmel 1986: 27 RS types in 5 dpf larvae, ~100-200 RS neurons/hemisphere.
N_RS       = 44    # named reticulospinal neurons — NEAR-EXACT (Hildebrand 2017)
N_ARF      = 400   # anterior reticular formation bilateral
N_IMRF     = 350   # intermediate reticular formation bilateral
N_PRF      = 350   # posterior reticular formation bilateral

# ── SPINAL CPG ───────────────────────────────────────────────────────────────
# Real: ~20-30 neurons/segment/side × ~30 segments × 2 sides = ~1,200-1,800.
# Ampatzis 2014 (J. Neurosci.): V2a (16-23), V0d, V1, dI6, MN (36-43) per hemi.
N_CPG_PER_SIDE = 75    # anterior-cord CPG per side (real: ~90-120 for first 5 segments)

# ── NEUROMODULATORY NUCLEI ────────────────────────────────────────────────────
# DA: posterior tuberculum DC2 (A11 homolog). NO VTA/SNc in zebrafish.
# NA: locus coeruleus A6 group (~20-40 bilateral neurons).
# 5HT: raphe (~244 total from imaging; ~81 DRN 5-HT neurons from projectome).
# ACh: habenula cholinergic + cranial motor co-release.
N_DA  = 150   # posterior tuberculum dopamine  (real: ~100-200)
N_NA  = 50    # locus coeruleus noradrenaline  (real: ~20-40 bilateral)
N_5HT = 200   # raphe serotonin                (real: ~244 from imaging)
N_ACH = 80    # acetylcholine                  (real: ~100 cholinergic habenula)

# ─────────────────────────────────────────────────────────────────────────────
# NEURON COUNT SUMMARY (~34K total, ~34% of real 100K larval zebrafish brain)
# ─────────────────────────────────────────────────────────────────────────────
# Region                 Neurons    Real (approx)    Source
# Retina (bilateral)      5,400    8,000            Kölsch 2021
# Optic tectum            8,400    35,000           Zabegalov 2022
# Pretectum               1,000    1,500            Estimate
# Thalamus + TRN          1,300    2,500            Bhatt 2012
# Cerebellum (all)        4,930    14,000           Knogler/Bhatt/Wullimann
# Inferior olive            300      300            Hsieh 2024  ← EXACT
# Habenula                1,200    1,500            Pandey 2018 ← near-exact
# Pallium + place cells   3,800    5,000            Cosacak 2023
# Basal ganglia           1,550    2,300            Estimate
# Olfactory bulb            600    1,000            Estimate
# Hypothalamus            2,000    4,000            Ryu 2020
# Raphe                     200      244            Gaspar & Bhatt 2014
# MON (lateral line)      1,000    1,500            2025 connectome
# Torus semicircularis      500    1,000            Estimate
# Reticular formation     1,144    2,000            Hildebrand 2017
# CPG (bilateral)           150    1,800            Ampatzis 2014
# Neuromodulatory           480      500            Various
# ─────────────────────────────────────────────────────────────────────────────
# Total                  ~34,000  ~100,000
# ─────────────────────────────────────────────────────────────────────────────
