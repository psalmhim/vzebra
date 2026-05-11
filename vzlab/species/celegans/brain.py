"""
CElegansNervousSystem — 302-neuron spiking network for C. elegans.

Implements the BrainModule ABC. Uses leaky integrate-and-fire dynamics
with the OpenWorm connectome weight matrix.

Tunable parameters (set via set_param / CLI --params):
  circuit.tau               Membrane time constant (default 20)
  circuit.electrical_weight Gap-junction weight scale (default 1.0)
  circuit.ei_ratio          E/I balance multiplier (default 1.0)
  motor.speed_base          Basal crawl speed (default 0.3)
  motor.reversal_threshold  AVA voltage to trigger reversal (default 0.8)
  sensory.adaptation_rate   AWC habituation rate per step (default 0.02)
  sensory.bilateral_diff_gain Left-right AWC difference gain (default 1.0)
  neuromod.serotonin_tone   Serotonin level: slows locomotion (default 0.0)
  chemotaxis.gain           Global chemical input gain (default 1.0)
  plasticity.hebbian_lr     Hebbian learning rate at AWC→AIY (default 0.0)

Ablatable regions:
  sensory, interneuron, motor        (bulk class ablations)
  AWC, AWB, AFD, AIY, AIZ, AIB,
  AVA, AVB, RIM, RIA, SMB, SMD      (named circuit nodes)
"""
from __future__ import annotations

import math
import numpy as np

from ...core.interfaces import BrainModule
from ...core.types import SensorySignals, MotorOutput, HierarchicalState
from .connectome import OpenWormConnectome, ALL_NEURONS, _NEURON_INDEX, _MOTOR


def _idx(*names: str) -> list[int]:
    return [_NEURON_INDEX[n] for n in names if n in _NEURON_INDEX]


class CElegansNervousSystem(BrainModule):

    TAU       = 20.0
    THRESHOLD = 1.0
    RESET     = 0.0
    DT        = 1.0

    def __init__(self, connectome: OpenWormConnectome | None = None):
        self._connectome = connectome or OpenWormConnectome.default()
        self._W  = self._connectome._W.copy()   # chemical synapses (302, 302)
        self._Wg = self._connectome._W.copy()   # gap-junction copy (scaled separately)
        self._N  = 302

        self._V      = np.zeros(self._N, dtype=np.float32)
        self._spikes = np.zeros(self._N, dtype=np.float32)
        self._spike_history: list[np.ndarray] = []

        # Sensory adaptation state for AWC neurons
        self._adaptation = np.zeros(self._N, dtype=np.float32)

        # Hebbian weight shadow for AWC→AIY synapses
        self._W_hebb = np.zeros((self._N, self._N), dtype=np.float32)

        self._ablated: set[str] = set()
        self._ablated_idx: set[int] = set()

        self._params: dict[str, float] = {
            "circuit.tau":                self.TAU,
            "circuit.electrical_weight":  1.0,
            "circuit.ei_ratio":           1.0,
            "motor.speed_base":           1.5,   # increased: ~0.3mm/s × 5s/step
            "motor.reversal_threshold":   0.8,
            "sensory.adaptation_rate":    0.02,
            "sensory.bilateral_diff_gain": 1.0,
            "circuit.klinokinesis_gain":  3.0,   # temporal gradient → AIZ modulation
            "neuromod.serotonin_tone":    0.0,
            "chemotaxis.gain":            1.0,
            "plasticity.hebbian_lr":      0.0,
        }

        self._prev_attr_conc: float = 0.0   # for klinokinesis (temporal gradient)
        self._pir_cooldown: int = 0          # prevent back-to-back pirouettes

        self._t = 0
        self._last_turn  = 0.0
        self._last_speed = 0.3
        self._last_goal  = 2   # EXPLORE default

        # ── Neuron indices ────────────────────────────────────────────────────
        self._awcl = _NEURON_INDEX.get("AWCL", 14)
        self._awcr = _NEURON_INDEX.get("AWCR", 15)
        self._awbl = _NEURON_INDEX.get("AWBL", 12)
        self._awbr = _NEURON_INDEX.get("AWBR", 13)

        # AFD: thermosensory (bilateral pair)
        self._afdl = _NEURON_INDEX.get("AFDL", 23)
        self._afdr = _NEURON_INDEX.get("AFDD", 20)   # AFDD is in list as "AFDD"

        self._aiy  = _NEURON_INDEX.get("AIY", 32)
        self._aiz  = _NEURON_INDEX.get("AIZ", 33)
        self._aib  = _NEURON_INDEX.get("AIB", 34)
        self._aia  = _NEURON_INDEX.get("AIA", 35)
        self._rim  = _NEURON_INDEX.get("RIM", 43)
        self._ria  = _NEURON_INDEX.get("RIA", 39)
        self._ava  = _NEURON_INDEX.get("AVA", 48)
        self._avb  = _NEURON_INDEX.get("AVB", 49)
        self._pvc  = _NEURON_INDEX.get("PVC", 62)

        # Motor pools
        self._va_idx = _idx(*[f"VA{i}" for i in range(1, 6)])
        self._db_idx = _idx(*[f"DB{i}" for i in range(1, 5)])
        self._vd_idx = _idx(*[f"VD{i}" for i in range(1, 5)])
        self._da_idx = _idx(*[f"DA{i}" for i in range(1, 5)])

        # SMB (smooth body bends — gentle turns)
        self._smb_idx = _idx("SMBDL", "SMBDR", "SMBVL", "SMBVR")
        # SMD (dorsal head muscles — sharp turns)
        self._smd_idx = _idx("SMDDL", "SMDDR", "SMDVL", "SMDVR")

        # Build named region map once
        self._region_map: dict[str, list[int]] = {
            "sensory":    list(range(32)),
            "interneuron": list(range(32, 108)),
            "motor":      list(range(108, 302)),
            "AWC":  [self._awcl, self._awcr],
            "AWB":  [self._awbl, self._awbr],
            "AFD":  [self._afdl, self._afdr],
            "AIY":  [self._aiy],
            "AIZ":  [self._aiz],
            "AIB":  [self._aib],
            "AVA":  [self._ava],
            "AVB":  [self._avb],
            "RIM":  [self._rim],
            "RIA":  [self._ria],
            "SMB":  self._smb_idx,
            "SMD":  self._smd_idx,
        }

    # ── BrainModule identity ──────────────────────────────────────────────────

    @property
    def species(self) -> str:
        return "c_elegans"

    @property
    def name(self) -> str:
        return "celegans_lif_302"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._V[:]           = 0.0
        self._spikes[:]      = 0.0
        self._adaptation[:]  = 0.0
        self._spike_history.clear()
        self._prev_attr_conc = 0.0
        self._pir_cooldown   = 0
        self._t          = 0
        self._last_turn  = 0.0
        self._last_speed = float(self._params["motor.speed_base"])
        self._last_goal  = 2

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, signals: SensorySignals, t: int) -> MotorOutput:
        self._t = t

        tau       = self._params["circuit.tau"]
        gain      = self._params["chemotaxis.gain"]
        ei_ratio  = self._params["circuit.ei_ratio"]
        gj_scale  = self._params["circuit.electrical_weight"]
        adapt_r   = self._params["sensory.adaptation_rate"]
        bdiff_g   = self._params["sensory.bilateral_diff_gain"]
        serotonin = self._params["neuromod.serotonin_tone"]
        hebb_lr   = self._params["plasticity.hebbian_lr"]
        rev_thr   = self._params["motor.reversal_threshold"]

        # ── External input ────────────────────────────────────────────────────
        I_ext = np.zeros(self._N, dtype=np.float32)

        if signals.chem is not None:
            attr_conc = float(signals.chem.concentrations[0]) * gain
            rep_conc  = (float(signals.chem.concentrations[1]) * gain
                         if len(signals.chem.concentrations) > 1 else 0.0)

            # Gate AWC-dependent chemosensory pathway on intact AWC neurons.
            awc_ablated = (self._awcl in self._ablated_idx or
                           self._awcr in self._ablated_idx)

            if not awc_ablated:
                # Sensory adaptation: sustained odor → AWC depression
                self._adaptation[self._awcl] += adapt_r * attr_conc
                self._adaptation[self._awcr] += adapt_r * attr_conc
                self._adaptation = np.clip(self._adaptation * 0.99, 0, 0.9)

                adapted_attr = attr_conc * (1.0 - self._adaptation[self._awcl])

                # AWC: bilateral-difference encodes gradient direction, scaled by
                # concentration — no bilateral drive when attractant is absent.
                grad = signals.chem.gradient
                if grad is not None and grad.ndim >= 2 and adapted_attr > 1e-9:
                    gx = float(grad[0, 0])
                    left_bias  = max(0.0, -gx) * bdiff_g
                    right_bias = max(0.0,  gx) * bdiff_g
                else:
                    left_bias = right_bias = 0.0

                I_ext[self._awcl] += adapted_attr + left_bias
                I_ext[self._awcr] += adapted_attr + right_bias

                # ── Klinokinesis: temporal gradient → pirouette rate ──────────
                # Rising dC/dt → suppress AIZ. Falling dC/dt → activate AIZ.
                klino = self._params.get("circuit.klinokinesis_gain", 3.0)
                if self._prev_attr_conc > 0:
                    dC_dt = attr_conc - self._prev_attr_conc
                    I_ext[self._aiz] += (-dC_dt) * klino
                self._prev_attr_conc = attr_conc

                # Direct forward drive: AVB receives concentration-scaled signal.
                I_ext[self._avb] += attr_conc * 0.5
            else:
                self._prev_attr_conc = 0.0   # no concentration memory when AWC absent

            # AWB (repellent) and tonic AIZ are independent of AWC.
            I_ext[self._awbl] += rep_conc * 0.8
            I_ext[self._awbr] += rep_conc * 0.8
            I_ext[self._aiz] += 0.20   # tonic baseline pirouette rate

        # Thermal input to AFD (signals.extras "temperature" if present)
        if signals.body is not None:
            temp = getattr(signals.body, 'temperature', None)
            if temp is not None:
                thermal_drive = float(np.clip(temp - 20.0, 0, 5) / 5.0)
                I_ext[self._afdl] += thermal_drive
                I_ext[self._afdr] += thermal_drive

            # Mechanosensory: wall contact → reversal reflex
            if signals.body.in_contact:
                I_ext[self._ava] += 0.6
                I_ext[self._pvc] += 0.4

        # ── E/I scaling ───────────────────────────────────────────────────────
        W_eff = self._W + self._W_hebb
        # Scale inhibitory synapses by ei_ratio (W<0 entries)
        W_mod = W_eff.copy()
        W_mod[W_mod < 0] *= ei_ratio
        W_mod[W_mod > 0] *= gj_scale   # gap-junction reuses positive weight scale

        # ── LIF dynamics ──────────────────────────────────────────────────────
        I_syn = W_mod.T @ self._spikes
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

        # ── Hebbian weight update at AWC → AIY ───────────────────────────────
        if hebb_lr > 0:
            pre  = self._spikes[[self._awcl, self._awcr]].mean()
            post = self._spikes[self._aiy]
            delta = hebb_lr * pre * post
            self._W_hebb[self._awcl, self._aiy] = np.clip(
                self._W_hebb[self._awcl, self._aiy] + delta, 0, 1.0
            )
            self._W_hebb[self._awcr, self._aiy] = np.clip(
                self._W_hebb[self._awcr, self._aiy] + delta, 0, 1.0
            )

        # ── Motor readout ─────────────────────────────────────────────────────
        fwd = float(self._spikes[self._db_idx].mean()) if self._db_idx else 0.0
        rev = float(self._spikes[self._va_idx].mean()) if self._va_idx else 0.0
        vd  = float(self._spikes[self._vd_idx].mean()) if self._vd_idx else 0.0

        # Pirouette: AIZ spiked or AVA above reversal threshold → sharp reorientation.
        # Check spikes (not post-reset voltage) to detect AIZ activation correctly.
        # Pirouette turn is NOT clipped here — the environment clips orientation change to
        # π/4 per step, so actual heading change is bounded; the unclipped turn value lets
        # Tier-1 assays correctly detect pirouettes via |turn| > π/4.
        aiz_spiked = bool(self._spikes[self._aiz])
        aiz_v = float(self._V[self._aiz])   # kept for L3 reporting
        # Cooldown prevents AVA-triggered double-pirouette the step after AIZ fires.
        self._pir_cooldown = max(0, self._pir_cooldown - 1)
        is_pirouette = (aiz_spiked or bool(self._spikes[self._ava])) and self._pir_cooldown == 0
        if is_pirouette:
            rng  = np.random.default_rng(t + int(self._V.sum() * 1000) % 10000)
            turn = float(rng.uniform(-math.pi, math.pi))
            self._pir_cooldown = 2   # block next 2 steps from re-triggering
        elif self._smb_idx:
            smb_act = float(self._spikes[self._smb_idx].mean())
            smd_act = float(self._spikes[self._smd_idx].mean()) if self._smd_idx else 0.0
            turn = float(np.clip((smb_act - smd_act) * 0.6 + (vd - 0.5) * 0.2,
                                 -math.pi / 4, math.pi / 4))
        else:
            turn = float(np.clip((vd - 0.5) * 0.4, -math.pi / 4, math.pi / 4))

        # Serotonin tone: slows locomotion, promotes dwelling
        sero_brake = 1.0 - serotonin * 0.5
        speed_base = self._params["motor.speed_base"] * sero_brake
        speed = max(0.0, float(fwd - rev * 0.6)) * speed_base
        if speed < 0.01:
            speed = speed_base * 0.4   # basal crawl

        # Goal: AIY > AIB → forage; AIB dominant → flee; else explore
        aiy_v = float(self._V[self._aiy])
        aib_v = float(self._V[self._aib])
        if aiy_v > aib_v + 0.15:
            self._last_goal = 0    # FORAGE
        elif aib_v > aiy_v + 0.2:
            self._last_goal = 1    # FLEE
        else:
            self._last_goal = 2    # EXPLORE

        self._last_turn  = float(turn)
        self._last_speed = float(np.clip(speed, 0.0, 1.0))
        return MotorOutput(agent_id=0, turn=self._last_turn, speed=self._last_speed)

    # ── Hierarchical state ────────────────────────────────────────────────────

    def get_hierarchical_state(self, t: int) -> HierarchicalState:
        recent = (np.stack(self._spike_history[-10:])
                  if self._spike_history else np.zeros((1, self._N)))
        mean_fr    = float(recent.mean())
        sensory_fr = float(recent[:, :32].mean())
        inter_fr   = float(recent[:, 32:108].mean())
        motor_fr   = float(recent[:, 108:].mean())

        afd_fr = float(recent[:, [self._afdl, self._afdr]].mean()) if self._spike_history else 0.0

        return HierarchicalState(
            t=t,
            agent_id=0,
            L1_synaptic={
                "mean_chem_weight":   float(np.abs(self._W).mean()),
                "hebb_total":         float(self._W_hebb.sum()),
                "adaptation_awcl":    float(self._adaptation[self._awcl]),
                "adaptation_awcr":    float(self._adaptation[self._awcr]),
                "ablated_neurons":    list(self._ablated),
            },
            L2_neuron={
                "mean_firing_rate":   mean_fr,
                "n_spikes_this_step": int(self._spikes.sum()),
                "membrane_v_mean":    float(self._V.mean()),
                "membrane_v_max":     float(self._V.max()),
            },
            L3_circuit={
                "sensory_fr":  sensory_fr,
                "inter_fr":    inter_fr,
                "motor_fr":    motor_fr,
                "afd_fr":      afd_fr,
                "awcl_v":      float(self._spikes[self._awcl]),
                "awcr_v":      float(self._spikes[self._awcr]),
                "aiy_v":       float(self._spikes[self._aiy]),
                "aiz_v":       float(self._spikes[self._aiz]),
                "aib_v":       float(self._spikes[self._aib]),
                "rim_v":       float(self._spikes[self._rim]),
                "ria_v":       float(self._spikes[self._ria]),
                "ava_v":       float(self._spikes[self._ava]),
                "avb_v":       float(self._spikes[self._avb]),
                "awcl_adapt":  float(self._adaptation[self._awcl]),
            },
            L4_region={
                "sensory_active":   sensory_fr > 0.05,
                "interneuron_active": inter_fr > 0.05,
                "motor_active":     motor_fr > 0.05,
                "afd_active":       afd_fr > 0.05,
                "ablated":          list(self._ablated),
            },
            L5_behaviour={
                "goal":            self._last_goal,
                "goal_name":       ["FORAGE", "FLEE", "EXPLORE"][self._last_goal],
                "turn":            self._last_turn,
                "speed":           self._last_speed,
                "serotonin_tone":  self._params["neuromod.serotonin_tone"],
                "chemotaxis_gain": self._params["chemotaxis.gain"],
                "adaptation_rate": self._params["sensory.adaptation_rate"],
            },
            L6_social={},  # C. elegans is largely solitary
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
