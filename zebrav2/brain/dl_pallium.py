"""
Spiking dorsolateral pallium (Dl): hippocampal homolog for spatial memory.

Zebrafish Dl (homologous to mammalian hippocampus):
  - Dorsolateral telencephalon, required for spatial learning
  - One-shot episodic memory: single-trial place-outcome associations
  - Pattern separation: sparse DG-like encoding distinguishes similar locations
  - Pattern completion: CA3-like recurrent attractor recalls from partial cues
  - Sharp-wave ripple (SWR) replay: offline consolidation during rest/sleep
  - Contextual remapping: different spatial maps for safe vs dangerous contexts

Distinct from ThetaPlaceCells: place_cells.py provides raw spatial activation
(Gaussian place fields + theta phase precession). Dl_pallium RECEIVES place
cell input and performs memory operations: encoding, storage, retrieval, replay.

Distinct from GeographicModel: geographic_model.py is a grid-based lookup table.
Dl_pallium learns associative spatial memories through spiking dynamics.

Free Energy Principle:
  Dl predicts spatial outcomes (food, risk) given current location+context.
  Prediction error drives one-shot encoding of novel place-outcome associations.
  SWR replay minimizes free energy by consolidating fragmented spatial memory.

Architecture:
  20 Izhikevich neurons:
    6 DG-like (RS, high threshold): pattern separation — sparse code
    8 CA3-like (RS, recurrent): pattern completion — attractor dynamics
    6 CA1-like (RS): output — context-dependent spatial memory readout
  + 4-channel TwoCompColumn (spatial, context, food memory, risk memory)
  + Episodic buffer: max 64 location-context-outcome triplets
  + Hebbian CA3 recurrent weights (STDP-like, one-shot)
"""
import math
import torch
import torch.nn as nn
import numpy as np
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingDlPallium(nn.Module):
    """Dorsolateral pallium: hippocampal spatial memory circuit."""

    def __init__(self, n_dg=6, n_ca3=8, n_ca1=6, max_episodes=64, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_dg = n_dg
        self.n_ca3 = n_ca3
        self.n_ca1 = n_ca1
        self.n = n_dg + n_ca3 + n_ca1  # 20 total

        # --- Spiking populations ---
        # DG: high threshold for sparse coding (pattern separation)
        self.dg = IzhikevichLayer(n_dg, 'RS', device)
        self.dg.i_tonic.fill_(-1.5)  # moderately quiet — fires on strong input

        # CA3: recurrent attractor (pattern completion)
        self.ca3 = IzhikevichLayer(n_ca3, 'RS', device)
        self.ca3.i_tonic.fill_(-0.5)  # easier to fire (attractor needs activity)

        # CA1: output layer (spatial memory readout)
        self.ca1 = IzhikevichLayer(n_ca1, 'RS', device)
        self.ca1.i_tonic.fill_(-0.5)

        # --- Synaptic weights ---
        # DG → CA3 (strong mossy fibers — "detonator" synapses, signed for spatial coding)
        # Fixed seed for reproducible spatial projections across runs
        _rng = torch.Generator(device='cpu')
        _rng.manual_seed(42)
        self.register_buffer('W_dg_ca3',
                             torch.randn(n_ca3, n_dg, generator=_rng).to(device) * 0.8)
        # CA3 → CA3 (recurrent — Schaffer collaterals / autoassociation)
        self.register_buffer('W_ca3_rec',
                             torch.zeros(n_ca3, n_ca3, device=device))
        # CA3 → CA1 (feedforward)
        self.register_buffer('W_ca3_ca1',
                             torch.randn(n_ca1, n_ca3, generator=_rng).abs().to(device) * 0.6)

        # --- Firing rate buffers ---
        self.register_buffer('rate_dg', torch.zeros(n_dg, device=device))
        self.register_buffer('rate_ca3', torch.zeros(n_ca3, device=device))
        self.register_buffer('rate_ca1', torch.zeros(n_ca1, device=device))
        self.register_buffer('rate', torch.zeros(self.n, device=device))

        # --- Episodic buffer ---
        self.max_episodes = max_episodes
        # Each episode: (place_code[n_ca3], context_vec[4], outcome[2])
        self._episodes_place = np.zeros((max_episodes, n_ca3), dtype=np.float32)
        self._episodes_context = np.zeros((max_episodes, 4), dtype=np.float32)
        self._episodes_outcome = np.zeros((max_episodes, 2), dtype=np.float32)  # [food, risk]
        self._episode_count = 0
        self._episode_idx = 0  # circular buffer pointer
        self._episode_strength = np.zeros(max_episodes, dtype=np.float32)

        # --- FEP: 4-channel prediction ---
        self.pc = TwoCompColumn(n_channels=4, n_per_ch=4, substeps=8, device=device)

        # --- State ---
        self.spatial_prediction = 0.0    # predicted outcome valence
        self.food_memory = 0.0           # recalled food association [0,1]
        self.risk_memory = 0.0           # recalled risk association [0,1]
        self.pattern_separation = 0.0    # DG sparsity (higher = more distinct)
        self.pattern_completion = 0.0    # CA3 attractor convergence [0,1]
        self.novelty = 0.0              # spatial novelty (no matching episode)
        self.replay_active = False       # SWR replay happening this step
        self._last_ca3_code = np.zeros(n_ca3, dtype=np.float32)
        self._theta_phase = 0.0
        self._steps_since_event = 0     # steps since last food/risk event

        # SWR replay parameters
        self._SWR_INTERVAL = 50         # replay every 50 quiet steps
        self._SWR_REPLAYS = 3           # episodes replayed per SWR event
        self._rest_counter = 0          # counts quiet steps

        # FEP state
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, place_activation: torch.Tensor,
                pos_x: float = 400.0, pos_y: float = 300.0,
                context: np.ndarray = None,
                food_eaten: bool = False, predator_near: bool = False,
                is_resting: bool = False, theta_phase: float = 0.0) -> dict:
        """
        place_activation: place cell rate vector [n_place_cells] from ThetaPlaceCells
        pos_x, pos_y: current position
        context: [4] vector (goal, energy_level, threat_level, time_of_day)
        food_eaten: one-shot encoding trigger for food
        predator_near: one-shot encoding trigger for risk
        is_resting: enables SWR replay
        theta_phase: from place cells, modulates encoding vs retrieval
        """
        self._theta_phase = theta_phase
        self._steps_since_event += 1

        if context is None:
            context = np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32)

        # --- 1. DG: Pattern Separation ---
        # Compress place cell input into sparse DG representation
        # Only top-k place cells drive DG (competitive inhibition)
        if place_activation.dim() == 0:
            place_activation = place_activation.unsqueeze(0)
        pc_input = place_activation[:min(len(place_activation), 128)]

        # Sparse projection: non-overlapping subsets for maximum discrimination
        n_pc = len(pc_input)
        dg_drive = torch.zeros(self.n_dg, device=self.device)
        chunk = max(1, n_pc // self.n_dg)
        for i in range(self.n_dg):
            # Each DG cell receives from a distinct subset of place cells
            start = i * chunk
            end = min(start + chunk, n_pc)
            dg_drive[i] = pc_input[start:end].sum() * 25.0

        # High threshold → only strongest inputs fire (pattern separation)
        for _ in range(8):
            self.dg(dg_drive + torch.randn(self.n_dg, device=self.device) * 0.2)
        self.rate_dg.copy_(self.dg.rate)

        # Sparsity metric
        dg_active = (self.rate_dg > 0.1).float()
        self.pattern_separation = 1.0 - float(dg_active.mean())  # higher = sparser

        # --- 2. CA3: Pattern Completion ---
        # Input: DG mossy fibers + CA3 recurrent (attractor)
        I_mossy = self.W_dg_ca3 @ self.rate_dg  # DG → CA3
        I_recurrent = self.W_ca3_rec @ self.rate_ca3  # CA3 → CA3 (from previous)

        # Theta modulation: encoding (peak) vs retrieval (trough)
        # At theta peak: DG input dominates (encoding new)
        # At theta trough: recurrent dominates (retrieving stored)
        theta_mod = math.sin(theta_phase)
        encode_weight = max(0, theta_mod) * 0.7 + 0.3   # 0.3–1.0
        retrieve_weight = max(0, -theta_mod) * 0.7 + 0.3  # 0.3–1.0

        I_ca3 = I_mossy * encode_weight * 15.0 + I_recurrent * retrieve_weight * 3.0

        for _ in range(20):
            self.ca3(I_ca3 + torch.randn(self.n_ca3, device=self.device) * 0.3)
        self.rate_ca3.copy_(self.ca3.rate)

        # Attractor convergence metric (compare CA3 activity across steps)
        ca3_np = self.rate_ca3.cpu().numpy()
        _ca3_norm = np.linalg.norm(ca3_np)
        if np.linalg.norm(self._last_ca3_code) > 1e-8 and _ca3_norm > 1e-8:
            cos_sim = np.dot(ca3_np, self._last_ca3_code) / (
                _ca3_norm * np.linalg.norm(self._last_ca3_code) + 1e-8)
            self.pattern_completion = float(max(0, cos_sim))
        self._last_ca3_code = ca3_np.copy()

        # --- 3. CA1: Output / Memory Readout ---
        I_ca1 = self.W_ca3_ca1 @ self.rate_ca3
        # Context gating: modulate CA1 by current context
        ctx_tensor = torch.tensor(context, device=self.device)
        ctx_gate = 1.0 + 0.3 * ctx_tensor.mean()  # mild context modulation

        for _ in range(8):
            self.ca1(I_ca1 * ctx_gate + torch.randn(self.n_ca1, device=self.device) * 0.2)
        self.rate_ca1.copy_(self.ca1.rate)

        # Combined rate
        self.rate[:self.n_dg] = self.rate_dg
        self.rate[self.n_dg:self.n_dg + self.n_ca3] = self.rate_ca3
        self.rate[self.n_dg + self.n_ca3:] = self.rate_ca1

        # --- 4. Memory Retrieval ---
        # Find most similar stored episode to current spatial code
        self.food_memory = 0.0
        self.risk_memory = 0.0
        self.novelty = 1.0  # assume novel until proven otherwise

        # Spatial fingerprint: deterministic DG drive projected to CA3-dim
        # (uses dg_drive which is deterministic from place cell input, unlike
        #  stochastic spiking rates which vary trial-to-trial)
        _query_code = self.W_dg_ca3.cpu().numpy() @ dg_drive.cpu().numpy()
        _qnorm = np.linalg.norm(_query_code)
        _query_code = _query_code / (_qnorm + 1e-8) if _qnorm > 1e-8 else _query_code

        if self._episode_count > 0:
            n = min(self._episode_count, self.max_episodes)
            stored = self._episodes_place[:n]
            # Cosine similarity with stored episodes
            stored_norm = stored / (np.linalg.norm(stored, axis=1, keepdims=True) + 1e-8)
            similarities = stored_norm @ _query_code  # [n_episodes]

            # Best match
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim > 0.4:  # threshold for retrieval
                self.novelty = max(0, 1.0 - best_sim)
                outcome = self._episodes_outcome[best_idx]
                # Weighted by similarity and episode strength
                strength = self._episode_strength[best_idx]
                self.food_memory = float(outcome[0]) * best_sim * strength
                self.risk_memory = float(outcome[1]) * best_sim * strength

        # Spatial prediction: net valence of recalled memories
        self.spatial_prediction = self.food_memory - self.risk_memory

        # --- 5. One-Shot Episodic Encoding ---
        # Salient events (food found, predator encountered) → immediate storage
        # Spatial code: deterministic DG drive projected through mossy fibers
        _spatial_code = self.W_dg_ca3.cpu().numpy() @ dg_drive.cpu().numpy()
        _sc_norm = np.linalg.norm(_spatial_code)
        _spatial_code = _spatial_code / (_sc_norm + 1e-8) if _sc_norm > 1e-8 else _spatial_code

        if food_eaten or predator_near:
            self._encode_episode(_spatial_code, context,
                                 food=1.0 if food_eaten else 0.0,
                                 risk=1.0 if predator_near else 0.0)
            # One-shot Hebbian: strengthen CA3 recurrent for this pattern
            self._strengthen_ca3_attractor(_spatial_code, eta=0.15)
            self._steps_since_event = 0

        # --- 6. Sharp-Wave Ripple Replay ---
        self.replay_active = False
        if is_resting or self._steps_since_event > self._SWR_INTERVAL:
            self._rest_counter += 1
            if self._rest_counter >= self._SWR_INTERVAL and self._episode_count > 0:
                self._swr_replay()
                self.replay_active = True
                self._rest_counter = 0
        else:
            self._rest_counter = 0

        # --- 7. FEP Prediction ---
        sensory = torch.tensor([
            float(place_activation.mean()),  # current spatial activation
            context[2],                       # threat context
            self.food_memory,                 # recalled food
            self.risk_memory,                 # recalled risk
        ], device=self.device, dtype=torch.float32)
        prediction = torch.tensor([
            float(place_activation.mean()),  # predict current location stable
            0.0,                              # predict safe
            self.food_memory,                 # predict food memory holds
            self.risk_memory,                 # predict risk memory holds
        ], device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'food_memory': self.food_memory,
            'risk_memory': self.risk_memory,
            'spatial_prediction': self.spatial_prediction,
            'novelty': self.novelty,
            'pattern_separation': self.pattern_separation,
            'pattern_completion': self.pattern_completion,
            'replay_active': self.replay_active,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
            'episode_count': self._episode_count,
        }

    def _encode_episode(self, ca3_code: np.ndarray, context: np.ndarray,
                        food: float, risk: float):
        """One-shot episodic encoding: store place-context-outcome triplet."""
        idx = self._episode_idx % self.max_episodes
        self._episodes_place[idx] = ca3_code
        self._episodes_context[idx] = context[:4]
        self._episodes_outcome[idx, 0] = food
        self._episodes_outcome[idx, 1] = risk
        self._episode_strength[idx] = 1.0  # fresh memory = full strength
        self._episode_idx += 1
        self._episode_count = min(self._episode_count + 1, self.max_episodes)

    def _strengthen_ca3_attractor(self, ca3_code: np.ndarray, eta: float = 0.1):
        """One-shot Hebbian: strengthen recurrent weights for active pattern."""
        # Outer product rule: w_ij += eta * r_i * r_j (autoassociation)
        code_t = torch.tensor(ca3_code, device=self.device)
        code_norm = code_t / (code_t.norm() + 1e-8)
        delta_W = eta * torch.outer(code_norm, code_norm)
        # Zero diagonal (no self-connections)
        delta_W.fill_diagonal_(0.0)
        self.W_ca3_rec.add_(delta_W)
        # Clip weights to prevent saturation
        self.W_ca3_rec.clamp_(-1.0, 1.0)

    def _swr_replay(self):
        """Sharp-wave ripple: replay stored episodes to consolidate."""
        if self._episode_count == 0:
            return
        n = min(self._episode_count, self.max_episodes)

        # Replay most recent episodes (recency bias)
        # Pick up to SWR_REPLAYS episodes
        n_replay = min(self._SWR_REPLAYS, n)
        # Recent episodes (circular buffer: most recent are near _episode_idx)
        indices = [(self._episode_idx - 1 - i) % self.max_episodes
                   for i in range(n_replay)]

        for idx in indices:
            stored_code = self._episodes_place[idx]
            if np.linalg.norm(stored_code) < 1e-6:
                continue
            # Replay: reactivate CA3 pattern and strengthen attractor
            self._strengthen_ca3_attractor(stored_code, eta=0.03)
            # Decay strength slightly (memories fade without rehearsal)
            self._episode_strength[idx] = min(1.0,
                                              self._episode_strength[idx] + 0.05)

        # Decay un-replayed episodes
        for i in range(n):
            if i not in indices:
                self._episode_strength[i] *= 0.995

    def get_efe_bias(self) -> dict:
        """Spatial memory bias on goal selection."""
        return {
            'forage': -self.food_memory * 0.2,   # food memory → attract forage
            'flee': -self.risk_memory * 0.3,     # risk memory → attract flee
            'explore': -self.novelty * 0.15,     # novelty → attract explore
        }

    def reset(self):
        self.dg.reset()
        self.ca3.reset()
        self.ca1.reset()
        self.rate_dg.zero_()
        self.rate_ca3.zero_()
        self.rate_ca1.zero_()
        self.rate.zero_()
        self.pc.reset()
        self.spatial_prediction = 0.0
        self.food_memory = 0.0
        self.risk_memory = 0.0
        self.pattern_separation = 0.0
        self.pattern_completion = 0.0
        self.novelty = 0.0
        self.replay_active = False
        self._last_ca3_code = np.zeros(self.n_ca3, dtype=np.float32)
        self._theta_phase = 0.0
        self._steps_since_event = 0
        self._rest_counter = 0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
        # Note: episodic buffer NOT cleared on reset (persistent memory)
