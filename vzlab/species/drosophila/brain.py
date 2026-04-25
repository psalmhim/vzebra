"""
DrosophilaBrain — Mushroom body olfactory conditioning circuit.

Implements BrainModule ABC with LIF dynamics (tau=20, threshold=1.0).

Key biology:
  Projection Neurons (PNs) relay odor identity from antennal lobe.
  Kenyon Cells (KCs) form sparse combinatorial representations.
  APL provides global feedback inhibition maintaining KC sparseness.
  DANs signal reward/punishment, gating KC→MBON synaptic plasticity.
  MBONs read out KC ensemble activity to guide approach/avoidance.

Tunable parameters:
  circuit.tau                Membrane time constant (default 20.0)
  circuit.kc_sparseness      Target fraction of KCs active (default 0.1)
  circuit.apl_strength       APL inhibition weight scale (default 2.0)
  plasticity.learning_rate   DAN-gated KC→MBON weight update rate (default 0.01)
  plasticity.decay_rate      Spontaneous forgetting rate (default 0.001)
  motor.approach_gain        MBON approach signal gain (default 1.0)
  motor.avoid_gain           MBON avoidance signal gain (default 1.0)
  neuromod.da_tone           Baseline dopamine (default 0.5)
  sensory.pn_gain            PN amplification of odor signals (default 1.0)

Ablatable regions:
  PN, KC_ab, KC_abp, KC_g, APL, DAN, MBON
"""
from __future__ import annotations

import math
import numpy as np

from ...core.interfaces import BrainModule
from ...core.types import SensorySignals, MotorOutput, HierarchicalState
from .connectome import DrosophilaConnectome, _N_TOTAL, _SLICES, _N_MBON, _N_DAN


class DrosophilaBrain(BrainModule):

    TAU       = 20.0
    THRESHOLD = 1.0
    RESET     = 0.0
    DT        = 1.0

    def __init__(self, connectome: DrosophilaConnectome | None = None):
        self._connectome = connectome or DrosophilaConnectome.default()
        self._N = _N_TOTAL

        # Slices for each population
        self._sl_pn    = _SLICES["PN"]
        self._sl_kc_ab = _SLICES["KC_ab"]
        self._sl_kc_ap = _SLICES["KC_abp"]
        self._sl_kc_g  = _SLICES["KC_g"]
        self._sl_apl   = _SLICES["APL"]
        self._sl_dan   = _SLICES["DAN"]
        self._sl_mbon  = _SLICES["MBON"]

        # All KC indices (contiguous: ab, abp, g)
        self._kc_start = self._sl_kc_ab.start
        self._kc_end   = self._sl_kc_g.stop
        self._kc_idx   = np.arange(self._kc_start, self._kc_end)

        # Partition MBONs: first half = approach, second half = avoidance
        mbon_mid = _N_MBON // 2
        self._mbon_approach_idx = np.arange(self._sl_mbon.start,
                                            self._sl_mbon.start + mbon_mid)
        self._mbon_avoid_idx    = np.arange(self._sl_mbon.start + mbon_mid,
                                            self._sl_mbon.stop)

        # Fixed structural weights from connectome (copy; W_plastic is learned)
        self._W_fixed   = self._connectome._W.copy()
        # Plastic KC→MBON submatrix (shape: n_kc_total × _N_MBON)
        n_kc = self._kc_end - self._kc_start
        self._W_plastic = self._connectome._W[self._kc_idx, :][:, self._sl_mbon].copy()

        # LIF state
        self._V      = np.zeros(self._N, dtype=np.float32)
        self._spikes = np.zeros(self._N, dtype=np.float32)
        self._spike_history: list[np.ndarray] = []

        # Ablation
        self._ablated: set[str] = set()
        self._ablated_idx: set[int] = set()

        self._params: dict[str, float] = {
            "circuit.tau":              self.TAU,
            "circuit.kc_sparseness":    0.1,
            "circuit.apl_strength":     2.0,
            "plasticity.learning_rate": 0.01,
            "plasticity.decay_rate":    0.001,
            "motor.approach_gain":      1.0,
            "motor.avoid_gain":         1.0,
            "neuromod.da_tone":         0.5,
            "sensory.pn_gain":          1.0,
        }

        self._t = 0
        self._last_turn  = 0.0
        self._last_speed = 0.5
        self._last_goal  = "EXPLORE"
        self._last_da    = 0.5

        # Random seed 42 for reproducibility of PN target assignment
        rng = np.random.default_rng(42)
        # CS+ drives first 25 PNs, CS- drives next 25 PNs
        pn_idx = np.arange(_SLICES["PN"].start, _SLICES["PN"].stop)
        self._cs_plus_pn  = pn_idx[:25]
        self._cs_minus_pn = pn_idx[25:]

        # Region map for ablation
        self._region_map: dict[str, list[int]] = {
            "PN":     list(range(self._sl_pn.start,    self._sl_pn.stop)),
            "KC_ab":  list(range(self._sl_kc_ab.start, self._sl_kc_ab.stop)),
            "KC_abp": list(range(self._sl_kc_ap.start, self._sl_kc_ap.stop)),
            "KC_g":   list(range(self._sl_kc_g.start,  self._sl_kc_g.stop)),
            "APL":    list(range(self._sl_apl.start,   self._sl_apl.stop)),
            "DAN":    list(range(self._sl_dan.start,   self._sl_dan.stop)),
            "MBON":   list(range(self._sl_mbon.start,  self._sl_mbon.stop)),
        }

    # ── BrainModule identity ──────────────────────────────────────────────────

    @property
    def species(self) -> str:
        return "drosophila_melanogaster"

    @property
    def name(self) -> str:
        return "drosophila_mushroom_body"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._V[:]      = 0.0
        self._spikes[:] = 0.0
        self._spike_history.clear()
        self._t         = 0
        self._last_turn  = 0.0
        self._last_speed = 0.5
        self._last_goal  = "EXPLORE"
        self._last_da    = float(self._params["neuromod.da_tone"])

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, signals: SensorySignals, t: int) -> MotorOutput:
        self._t = t

        tau      = self._params["circuit.tau"]
        sparsity = self._params["circuit.kc_sparseness"]
        apl_str  = self._params["circuit.apl_strength"]
        lr       = self._params["plasticity.learning_rate"]
        decay    = self._params["plasticity.decay_rate"]
        app_gain = self._params["motor.approach_gain"]
        avo_gain = self._params["motor.avoid_gain"]
        da_tone  = self._params["neuromod.da_tone"]
        pn_gain  = self._params["sensory.pn_gain"]

        # ── External input (PN drive from odor concentrations) ────────────────
        I_ext = np.zeros(self._N, dtype=np.float32)

        cs_plus_conc  = 0.0
        cs_minus_conc = 0.0
        if signals.chem is not None and len(signals.chem.concentrations) >= 2:
            cs_plus_conc  = float(signals.chem.concentrations[0]) * pn_gain
            cs_minus_conc = float(signals.chem.concentrations[1]) * pn_gain

        if "PN" not in self._ablated:
            I_ext[self._cs_plus_pn]  += cs_plus_conc
            I_ext[self._cs_minus_pn] += cs_minus_conc

        # ── DAN activity (reward/punishment) ─────────────────────────────────
        reward     = 0.0
        punishment = 0.0
        if signals.extras:
            reward = float(signals.extras.get("reward", 0.0))
        if signals.body is not None and signals.body.in_contact:
            punishment = 1.0

        da_signal = da_tone + reward - punishment * 0.5
        da_signal = float(np.clip(da_signal, 0.0, 2.0))
        self._last_da = da_signal

        if "DAN" not in self._ablated:
            I_ext[self._sl_dan.start:self._sl_dan.stop] += da_signal * 0.3

        # ── Build effective weight matrix ─────────────────────────────────────
        # Inject plastic KC→MBON weights back into the full weight matrix
        W_eff = self._W_fixed.copy()
        W_eff[np.ix_(self._kc_idx, np.arange(self._sl_mbon.start, self._sl_mbon.stop))] = \
            self._W_plastic

        # ── LIF dynamics ──────────────────────────────────────────────────────
        I_syn = W_eff.T @ self._spikes
        dV    = (-self._V / tau + I_syn + I_ext) * self.DT
        self._V += dV

        # Zero ablated neurons
        for idx in self._ablated_idx:
            self._V[idx] = 0.0

        # Spiking
        fired        = self._V >= self.THRESHOLD
        self._spikes = fired.astype(np.float32)
        self._V[fired] = self.RESET
        self._spike_history.append(self._spikes.copy())

        # ── APL: global winner-take-all inhibition on KCs ─────────────────────
        # Suppress excess KCs so at most sparsity*n_kc remain active.
        # Binary spikes require counting active KCs directly rather than
        # using a voltage threshold (which is always 1.0 with LIF resets).
        if "APL" not in self._ablated:
            kc_activity = self._spikes[self._kc_idx]
            n_kc = len(self._kc_idx)
            n_keep = max(1, int(sparsity * n_kc))
            active_local = np.where(kc_activity > 0)[0]
            if len(active_local) > n_keep:
                suppress_local = active_local[n_keep:]
                suppress_idx = self._kc_idx[suppress_local]
                self._spikes[suppress_idx] = 0.0
                self._V[suppress_idx]      = 0.0

        # ── DAN-gated KC→MBON plasticity ─────────────────────────────────────
        if "DAN" not in self._ablated and "MBON" not in self._ablated:
            kc_fr   = self._spikes[self._kc_idx]       # (n_kc,)
            mbon_fr = self._spikes[self._sl_mbon.start:self._sl_mbon.stop]  # (_N_MBON,)

            if reward > 0:
                # Strengthen KC→MBON_approach (first half of MBONs)
                half = _N_MBON // 2
                delta_app = lr * da_signal * np.outer(kc_fr, mbon_fr[:half])
                self._W_plastic[:, :half] = np.clip(
                    self._W_plastic[:, :half] + delta_app, 0.0, 2.0
                )

            if punishment > 0:
                # Weaken KC→MBON_approach (anti-Hebbian)
                half = _N_MBON // 2
                delta_pun = lr * np.outer(kc_fr, mbon_fr[:half])
                self._W_plastic[:, :half] = np.clip(
                    self._W_plastic[:, :half] - delta_pun, 0.0, 2.0
                )

            # Spontaneous decay
            self._W_plastic = np.clip(
                self._W_plastic * (1.0 - decay), 0.0, 2.0
            )

        # ── Motor readout from MBONs ──────────────────────────────────────────
        mbon_spikes = self._spikes[self._sl_mbon.start:self._sl_mbon.stop]
        half = _N_MBON // 2
        mbon_approach = float(mbon_spikes[:half].mean()) * app_gain
        mbon_avoid    = float(mbon_spikes[half:].mean()) * avo_gain

        # Turn: asymmetric MBON activity → left/right bias
        # Use alternating L/R assignment within each MBON half
        mbon_left  = float(mbon_spikes[0::2].mean())
        mbon_right = float(mbon_spikes[1::2].mean())
        turn = float(np.clip((mbon_right - mbon_left) * 2.0, -math.pi / 4, math.pi / 4))

        # Speed: approach MBONs drive forward; avoidance MBONs suppress
        approach_drive = mbon_approach - mbon_avoid * 0.5
        speed = float(np.clip(0.5 + approach_drive, 0.1, 1.0))

        # Goal classification
        if mbon_approach > mbon_avoid + 0.02:
            self._last_goal = "APPROACH"
        elif mbon_avoid > mbon_approach + 0.02:
            self._last_goal = "AVOID"
        else:
            self._last_goal = "EXPLORE"

        self._last_turn  = turn
        self._last_speed = speed
        return MotorOutput(agent_id=0, turn=turn, speed=speed)

    # ── Hierarchical state ────────────────────────────────────────────────────

    def get_hierarchical_state(self, t: int) -> HierarchicalState:
        recent = (np.stack(self._spike_history[-10:])
                  if len(self._spike_history) >= 1
                  else np.zeros((1, self._N)))

        mean_fr   = float(recent.mean())
        n_spikes  = int(self._spikes.sum())

        kc_act    = self._spikes[self._kc_idx]
        n_kc      = len(self._kc_idx)
        n_active_kcs = int(kc_act.sum())
        kc_sparseness_actual = float(n_active_kcs / max(1, n_kc))

        pn_fr  = float(self._spikes[self._sl_pn.start:self._sl_pn.stop].mean())
        kc_fr  = float(kc_act.mean())
        apl_v  = float(self._V[self._sl_apl.start])
        dan_act = float(self._spikes[self._sl_dan.start:self._sl_dan.stop].mean())

        half = _N_MBON // 2
        mbon_spikes = self._spikes[self._sl_mbon.start:self._sl_mbon.stop]
        mbon_approach = float(mbon_spikes[:half].mean())
        mbon_avoid    = float(mbon_spikes[half:].mean())

        mean_kc_mbon_weight = float(self._W_plastic.mean())

        return HierarchicalState(
            t=t,
            agent_id=0,
            L1_synaptic={
                "mean_kc_mbon_weight": mean_kc_mbon_weight,
                "n_active_KCs":        n_active_kcs,
            },
            L2_neuron={
                "mean_firing_rate": mean_fr,
                "n_spikes":         n_spikes,
                "kc_sparseness":    kc_sparseness_actual,
            },
            L3_circuit={
                "pn_fr":       pn_fr,
                "kc_fr":       kc_fr,
                "apl_v":       apl_v,
                "mbon_approach": mbon_approach,
                "mbon_avoid":  mbon_avoid,
                "dan_activity": dan_act,
            },
            L4_region={
                "PN_active":    pn_fr > 0.05,
                "KC_ab_active": float(self._spikes[self._sl_kc_ab.start:self._sl_kc_ab.stop].mean()) > 0.05,
                "KC_abp_active": float(self._spikes[self._sl_kc_ap.start:self._sl_kc_ap.stop].mean()) > 0.05,
                "KC_g_active":  float(self._spikes[self._sl_kc_g.start:self._sl_kc_g.stop].mean()) > 0.05,
                "APL_active":   apl_v > 0.05,
                "DAN_active":   dan_act > 0.05,
                "MBON_active":  (mbon_approach + mbon_avoid) > 0.05,
                "ablated":      list(self._ablated),
            },
            L5_behaviour={
                "goal_name":     self._last_goal,
                "turn":          self._last_turn,
                "speed":         self._last_speed,
                "da_tone":       float(self._params["neuromod.da_tone"]),
                "learning_rate": float(self._params["plasticity.learning_rate"]),
            },
            L6_social={},  # Drosophila is solitary in this model
        )

    # ── Intervention API ──────────────────────────────────────────────────────

    def ablate(self, region: str) -> None:
        self._ablated.add(region)
        for idx in self._region_map.get(region, []):
            self._ablated_idx.add(idx)

    def restore(self, region: str) -> None:
        self._ablated.discard(region)
        for idx in self._region_map.get(region, []):
            self._ablated_idx.discard(idx)

    def set_param(self, path: str, value: float) -> None:
        self._params[path] = value

    def get_param(self, path: str) -> float:
        return self._params.get(path, 0.0)

    def list_regions(self) -> list[str]:
        return list(self._region_map.keys())

    def list_params(self) -> list[str]:
        return list(self._params.keys())
