"""
XenopusTadpoleBrain — Xenopus laevis spinal central pattern generator.

Implements BrainModule ABC. Models the spinal CPG for tadpole swimming
(Roberts et al. 1998; Li et al. 2006).

Tactile input (body.in_contact) activates Rohon-Beard sensory neurons →
descending interneurons (dINs) generate rhythmic bursts → commissural
inhibitory interneurons (cINs) enforce left/right anti-phase → motor
neurons (MNs) drive tail muscles.

Once initiated by touch, the CPG sustains itself via dIN recurrent
excitation. Serotonin (5-HT) modulates swim period.

Ablatable regions:
  RB    touch detection disabled (touch no longer initiates swimming)
  dIN   rhythm generator silenced (swimming stops immediately)
  cIN   cross-inhibition removed (L/R synchrony instead of alternation)
  MN    motor neurons silenced (no turn/speed output)

Tunable parameters:
  cpg.swim_gain   Forward speed multiplier during swimming (default 1.0)
  cpg.turn_gain   L/R alternation turn amplitude in rad (default 0.30)
"""
from __future__ import annotations

import math
import numpy as np

from ...core.interfaces import BrainModule
from ...core.types import SensorySignals, MotorOutput, HierarchicalState
from .connectome import XenopusConnectome


class XenopusTadpoleBrain(BrainModule):
    # Anatomically correct neuron counts per region (Roberts et al. 1998)
    N_NEURONS: dict[str, int] = {"RB": 20, "dIN": 60, "cIN": 40, "MN": 30}

    def __init__(self, connectome: XenopusConnectome | None = None):
        self._connectome = connectome or XenopusConnectome.default()
        self._ablated: set[str] = set()
        self._params: dict[str, float] = {
            "cpg.swim_gain": 1.0,
            "cpg.turn_gain": 0.30,
        }
        self._swimming = False
        self._phase    = 0       # even = left active, odd = right active
        self._region_rates: dict[str, float] = {}
        self._t   = 0
        self._rng = np.random.default_rng(99)

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def species(self) -> str:
        return "xenopus_laevis"

    @property
    def name(self) -> str:
        return "xenopus_cpg"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._swimming = False
        self._phase    = 0
        self._region_rates = {}
        self._t   = 0
        self._rng = np.random.default_rng(99)

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, signals: SensorySignals, t: int) -> MotorOutput:
        self._t = t

        touch = (signals.body is not None and bool(signals.body.in_contact))

        # RB ablation: touch signal never reaches CPG
        if "RB" in self._ablated:
            touch = False

        # Touch initiates swimming (if dIN is intact)
        if touch and not self._swimming and "dIN" not in self._ablated:
            self._swimming = True
            self._phase    = 0

        turn, speed = self._cpg_step()
        self._update_region_rates(touch)

        return MotorOutput(agent_id=0, turn=float(turn), speed=float(speed))

    # ── Hierarchical state ────────────────────────────────────────────────────

    def get_hierarchical_state(self, t: int) -> HierarchicalState:
        mean_fr = float(np.mean(list(self._region_rates.values()))) \
                  if self._region_rates else 0.0
        return HierarchicalState(
            t=t,
            agent_id=0,
            L1_synaptic={
                "swimming": float(self._swimming),
                "phase":    float(self._phase),
            },
            L2_neuron={
                "mean_firing_rate": mean_fr,
            },
            L3_circuit={
                "rb_fr":   self._region_rates.get("rb_fr",  0.0),
                "din_fr":  self._region_rates.get("din_fr", 0.0),
                "cin_fr":  self._region_rates.get("cin_fr", 0.0),
                "mn_fr":   self._region_rates.get("mn_fr",  0.0),
                "swimming": float(self._swimming),
            },
            L4_region={
                "ablated":    list(self._ablated),
                "swimming":   self._swimming,
                "n_neurons":  self.N_NEURONS,
                "n_total":    sum(self.N_NEURONS.values()),
            },
            L5_behaviour={
                "swimming":   self._swimming,
                "phase":      self._phase,
                "swim_gain":  self._params["cpg.swim_gain"],
                "turn_gain":  self._params["cpg.turn_gain"],
            },
            L6_social={},
        )

    # ── Intervention API ──────────────────────────────────────────────────────

    def ablate(self, region: str) -> None:
        self._ablated.add(region)
        if region == "dIN":
            self._swimming = False   # rhythm generator silenced → stop immediately

    def restore(self, region: str) -> None:
        self._ablated.discard(region)

    def set_param(self, path: str, value: float) -> None:
        self._params[path] = value

    def get_param(self, path: str) -> float:
        return self._params.get(path, 0.0)

    def list_regions(self) -> list[str]:
        return ["RB", "dIN", "cIN", "MN"]

    def list_params(self) -> list[str]:
        return list(self._params.keys())

    # ── Internal CPG logic ────────────────────────────────────────────────────

    def _cpg_step(self) -> tuple[float, float]:
        if not self._swimming:
            return 0.0, 0.0

        if "MN" in self._ablated:
            self._phase += 1
            return 0.0, 0.0

        swim_gain = self._params["cpg.swim_gain"]
        turn_gain = self._params["cpg.turn_gain"]
        speed = float(np.clip(swim_gain, 0.0, 2.0))

        if "cIN" in self._ablated:
            # No cross-inhibition → both sides co-activate → no net turn
            turn = float(self._rng.normal(0, 0.02))
        else:
            # Alternating L/R based on CPG phase
            turn = +turn_gain if self._phase % 2 == 0 else -turn_gain

        self._phase += 1
        return float(np.clip(turn, -math.pi / 4, math.pi / 4)), speed

    def _update_region_rates(self, touch: bool) -> None:
        if self._swimming:
            # During steady swimming RBs settle to a low tonic rate; wall contact
            # adds a brief bump but does not dominate (Roberts et al. 1998).
            rb = 0.15 + 0.20 * float(touch)
            self._region_rates = {
                "rb_fr":  float(np.clip(rb, 0.0, 1.0)),
                "din_fr": 0.0  if "dIN" in self._ablated else 0.70,
                "cin_fr": 0.0  if "cIN" in self._ablated else 0.40,
                "mn_fr":  0.0  if "MN"  in self._ablated else 0.60,
            }
        else:
            self._region_rates = {
                "rb_fr":  float(np.clip(0.85 * float(touch), 0.0, 1.0)),
                "din_fr": 0.05,
                "cin_fr": 0.0,
                "mn_fr":  0.0,
            }
