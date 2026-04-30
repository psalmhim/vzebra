"""
Spiking pituitary gland: master endocrine organ.

Zebrafish pituitary (adenohypophysis + neurohypophysis):
  - Anterior pituitary (adenohypophysis):
    ACTH (corticotrophs): stress response via HPI axis → interrenal gland
    TSH (thyrotrophs): thyroid-stimulating hormone → metabolism
    GH (somatotrophs): growth hormone → somatic growth
  - Intermediate lobe (pars intermedia):
    MSH/α-MSH (melanotrophs): melanophore-stimulating hormone → skin darkening
      Zebrafish-specific: rapid background adaptation via melanophore aggregation
      Tonically inhibited by hypothalamic dopamine (D2 receptors)
    β-endorphin (co-released with MSH from POMC): pain modulation, reward
  - Posterior pituitary (neurohypophysis):
    Vasotocin (AVT, fish AVP homolog): osmoregulation, social dominance
      Zebrafish AVT neurons in preoptic area project to pituitary
    Isotocin (IT, fish oxytocin homolog): social affiliation, shoaling
      Isotocin promotes prosocial behavior in zebrafish

The hypothalamic-pituitary-interrenal (HPI) axis in zebrafish:
  CRH (hypothalamus) → ACTH (pituitary) → cortisol (interrenal gland)
  Cortisol provides negative feedback to both hypothalamus and pituitary.

Free Energy Principle:
  Pituitary implements allostatic regulation: hormonal outputs are active
  inference actions that minimize expected free energy of bodily states.
  Prediction errors in homeostatic variables drive hormonal release to
  restore setpoints.  Precision = metabolic urgency.

Architecture:
  12 RS neurons: 4 anterior (ACTH/TSH/GH), 4 intermediate (MSH/endorphin),
                 4 posterior (vasotocin/isotocin)
  + 3-channel TwoCompColumn (anterior PE, intermediate PE, posterior PE)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingPituitary(nn.Module):
    def __init__(self, n_neurons=12, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-0.8)  # moderate baseline; responsive to CRH
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 3 channels (anterior PE, intermediate PE, posterior PE)
        self.pc = TwoCompColumn(n_channels=3, n_per_ch=4, substeps=8, device=device)

        # Hormonal output state
        self.acth = 0.0              # ACTH: stress hormone precursor [0,1]
        self.cortisol_drive = 0.0    # ACTH→interrenal → cortisol [0,1]
        self.msh = 0.0               # MSH: melanophore stimulating [0,1]
        self.endorphin = 0.0         # β-endorphin: pain/reward [0,1]
        self.vasotocin = 0.0         # fish AVP homolog [0,1]
        self.isotocin = 0.0          # fish oxytocin homolog [0,1]

        # Internal processing
        self._tsh = 0.0              # thyroid-stimulating hormone
        self._gh = 0.0               # growth hormone
        self._pomc_activity = 0.0    # POMC precursor (→ ACTH + MSH + endorphin)
        self._da_inhibition_coeff = 0.7  # D2 dopamine inhibition on MSH (disorder-modulatable)

        # Setpoints
        self._stress_setpoint = 0.1  # expected low stress
        self._energy_setpoint = 60.0 # expected adequate energy
        self._social_setpoint = 0.3  # expected moderate social contact

        # FEP state
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, crh_release: float = 0.0,
                stress: float = 0.0,
                energy: float = 50.0,
                melatonin: float = 0.0,
                dopamine: float = 0.5) -> dict:
        """
        crh_release: CRH from hypothalamus PVN [0, 1] (drives ACTH)
        stress: allostatic stress level [0, 1]
        energy: current energy level [0, 100]
        melatonin: circadian melatonin [0, 1]
        dopamine: tonic dopamine from hypothalamus [0, 1]
        """
        # --- POMC processing ---
        # POMC is the precursor peptide: cleaved into ACTH, MSH, β-endorphin
        # High CRH preferentially drives ACTH pathway
        self._pomc_activity = min(1.0, crh_release * 0.6 + stress * 0.3)

        # --- Anterior lobe: ACTH, TSH, GH ---
        # ACTH: driven by CRH, amplified by stress
        self.acth = min(1.0, crh_release * 0.7 + stress * 0.3)
        # Cortisol drive: ACTH → interrenal gland → cortisol
        self.cortisol_drive = min(1.0, self.acth * 0.8)

        # TSH: metabolic demand, suppressed by stress (sick-euthyroid)
        energy_norm = max(0.0, min(1.0, energy / 100.0))
        self._tsh = max(0.0, min(1.0, (1.0 - energy_norm) * 0.5 * (1.0 - stress * 0.3)))

        # GH: growth when energy is adequate and stress is low
        self._gh = max(0.0, min(1.0, energy_norm * 0.4 * (1.0 - stress * 0.5)))

        # --- Intermediate lobe: MSH, endorphin ---
        # MSH: melanophore stimulating hormone
        # Tonically INHIBITED by hypothalamic dopamine (D2 receptor)
        # Low dopamine → disinhibition → MSH release → skin darkening
        dopamine_inhibition = dopamine * self._da_inhibition_coeff  # tonic inhibition
        raw_msh = self._pomc_activity * 0.5 * (1.0 - dopamine_inhibition)
        # Stress and melatonin also promote MSH (background adaptation)
        raw_msh += stress * 0.2 + melatonin * 0.1
        self.msh = min(1.0, max(0.0, raw_msh))

        # β-endorphin: co-released with MSH from POMC
        # Analgesic + reward; high during stress (stress-induced analgesia)
        self.endorphin = min(1.0, self._pomc_activity * 0.4 + stress * 0.2)

        # --- Posterior lobe: vasotocin, isotocin ---
        # Vasotocin (AVT): osmoregulation + social dominance
        # Released under osmotic stress and social challenge
        self.vasotocin = min(1.0, stress * 0.3 + (1.0 - energy_norm) * 0.2 + 0.1)

        # Isotocin (IT): social affiliation, shoaling
        # Promoted by low stress, suppressed by high stress
        raw_it = (1.0 - stress) * 0.5 + (1.0 - melatonin) * 0.2
        self.isotocin = min(1.0, max(0.0, raw_it))

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        # Anterior neurons (0-3)
        I[0] = self.acth * 15.0           # ACTH corticotroph
        I[1] = self.cortisol_drive * 12.0 # cortisol drive
        I[2] = self._tsh * 10.0          # TSH thyrotroph
        I[3] = self._gh * 8.0            # GH somatotroph
        # Intermediate neurons (4-7)
        I[4] = self.msh * 12.0           # MSH melanotroph
        I[5] = (1.0 - dopamine) * 10.0   # dopamine disinhibition
        I[6] = self.endorphin * 12.0     # β-endorphin
        I[7] = self._pomc_activity * 8.0 # POMC precursor
        # Posterior neurons (8-11)
        I[8] = self.vasotocin * 12.0     # vasotocin (AVT)
        I[9] = stress * 10.0             # osmotic/social stress
        I[10] = self.isotocin * 12.0     # isotocin (IT)
        I[11] = (1.0 - stress) * 8.0     # social calm signal

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        # Channel 0: anterior PE (stress response adequacy)
        # Channel 1: intermediate PE (POMC/MSH regulation)
        # Channel 2: posterior PE (osmotic/social balance)
        sensory = torch.tensor([
            stress, self._pomc_activity, self.vasotocin
        ], device=self.device, dtype=torch.float32)
        prediction = torch.tensor([
            self._stress_setpoint,
            0.2,  # expected low POMC at rest
            0.2,  # expected low vasotocin at rest
        ], device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'acth': self.acth,
            'cortisol_drive': self.cortisol_drive,
            'msh': self.msh,
            'endorphin': self.endorphin,
            'vasotocin': self.vasotocin,
            'isotocin': self.isotocin,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.acth = 0.0
        self.cortisol_drive = 0.0
        self.msh = 0.0
        self.endorphin = 0.0
        self.vasotocin = 0.0
        self.isotocin = 0.0
        self._tsh = 0.0
        self._gh = 0.0
        self._pomc_activity = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
