"""
VAE World Model for Active Inference Planning (Step 16b — multi-modal).

Encodes fused tectal output (oF) *plus* a 13-dim multi-modal state context
into a latent space z, learns a transition model for action-conditioned
state prediction, and uses associative memory + forward simulation to
produce planning bonuses for goal selection.

The latent z now serves as a proper active-inference belief state μ — the
sufficient statistic of the agent's approximate posterior over hidden causes.
The 13-dim state context captures four modalities:

  Proprioceptive [3]: x, y (normalised position), heading / π
  Interoceptive  [1]: energy / 100
  Neuromodulatory[2]: dopamine, RPE (clipped)
  Precision      [2]: π_OT, π_PC
  Exteroceptive  [3]: classifier top-3 (food, enemy, environ)
  Cognitive      [2]: cross-modal surprise, visual free energy

Encoder input = pooled oF (64) + state context (13) = 77 dim.
Decoder reconstructs only the pooled oF (64) — state context is
conditioning information, not a prediction target.

Components:
  - VAEEncoder / VAEDecoder: compress [pooled oF ‖ state_ctx] (77) → z (16)
  - TransitionModel: z' = z + 0.3 * delta(z, action_ctx)
  - AssociativeMemory: 64 RBF nodes mapping z → (food_rate, risk)
  - VAEWorldModel: orchestrator with online ELBO training and EFE planning

~8,600 parameters. Trains online each step via ring-buffer mini-batches.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """Encode [pooled oF ‖ state_ctx] (77-dim) → latent mu, logvar (16-dim)."""

    def __init__(self, in_dim=77, hidden=48, latent=16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    """Decode latent z (16-dim) → reconstructed pooled oF (64-dim)."""

    def __init__(self, latent=16, hidden=48, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc_out(h)


class TransitionModel(nn.Module):
    """Predict next latent state: z' = z + 0.3 * delta(z, action_ctx).

    action_ctx: [turn_rate, speed, goal_onehot[3]] = 5-dim
    """

    RESIDUAL_SCALE = 0.3

    def __init__(self, latent=16, act_dim=5, hidden=24):
        super().__init__()
        self.fc1 = nn.Linear(latent + act_dim, hidden)
        self.fc_out = nn.Linear(hidden, latent)

    def forward(self, z, action_ctx):
        """
        Args:
            z: [B, 16]
            action_ctx: [B, 5]
        Returns:
            z_next: [B, 16]
        """
        h = F.relu(self.fc1(torch.cat([z, action_ctx], dim=-1)))
        delta = torch.tanh(self.fc_out(h))
        return z + self.RESIDUAL_SCALE * delta


# ---------------------------------------------------------------------------
# Associative memory (pure numpy — no gradients)
# ---------------------------------------------------------------------------

class AssociativeMemory:
    """RBF kernel memory mapping latent z → (food_rate, risk).

    64 nodes, each storing (centroid[16], food_rate, risk, visit_count).
    Hebbian-style online updates with centroid drift.
    """

    def __init__(self, n_nodes=64, latent_dim=16, kernel_width=2.0,
                 match_threshold=3.0, ema_rate=0.1, drift_rate=0.02):
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.kernel_width = kernel_width
        self.match_threshold = match_threshold
        self.ema_rate = ema_rate
        self.drift_rate = drift_rate

        self.centroids = np.zeros((n_nodes, latent_dim), dtype=np.float32)
        self.food_rate = np.zeros(n_nodes, dtype=np.float32)
        self.risk = np.zeros(n_nodes, dtype=np.float32)
        self.visit_count = np.zeros(n_nodes, dtype=np.float32)
        self.n_allocated = 0

    def _rbf_weights(self, z):
        """Compute RBF weights for query point z against all allocated nodes."""
        if self.n_allocated == 0:
            return np.zeros(self.n_nodes, dtype=np.float32)
        active = self.centroids[:self.n_allocated]
        diff = active - z[np.newaxis, :]
        dist_sq = np.sum(diff ** 2, axis=1)
        w = np.exp(-dist_sq / (2.0 * self.kernel_width ** 2))
        full_w = np.zeros(self.n_nodes, dtype=np.float32)
        full_w[:self.n_allocated] = w
        return full_w

    def query(self, z):
        """Query memory at latent point z.

        Returns:
            food_rate: float — expected food availability
            risk: float — expected danger level
        """
        w = self._rbf_weights(z)
        w_sum = w.sum() + 1e-8
        food = float(np.dot(w, self.food_rate) / w_sum)
        risk_val = float(np.dot(w, self.risk) / w_sum)
        return food, risk_val

    def query_epistemic(self, z):
        """Query epistemic signals at latent point z.

        Returns:
            novelty: float [0, 1] — high when region is unvisited
            confidence: float [0, 1] — max RBF weight (memory match quality)
        """
        if self.n_allocated == 0:
            return 1.0, 0.0
        w = self._rbf_weights(z)
        w_sum = w.sum() + 1e-8
        raw_familiarity = float(np.dot(w, self.visit_count) / w_sum)
        familiarity = raw_familiarity / (raw_familiarity + 5.0)  # half-max at 5 visits
        novelty = 1.0 - familiarity
        max_w = float(w[:self.n_allocated].max())
        return novelty, max_w

    def update(self, z, food_signal, risk_signal):
        """Update memory with observed (food, risk) at latent z.

        Allocates a new node if no existing node is close enough.
        Recycles least-visited node when full.
        """
        w = self._rbf_weights(z)
        best_idx = int(np.argmax(w[:max(self.n_allocated, 1)]))
        best_w = w[best_idx] if self.n_allocated > 0 else 0.0

        if best_w < np.exp(-self.match_threshold ** 2 / 2.0):
            # No close match — allocate or recycle
            if self.n_allocated < self.n_nodes:
                idx = self.n_allocated
                self.n_allocated += 1
            else:
                # Recycle least-visited node
                idx = int(np.argmin(self.visit_count))
            self.centroids[idx] = z.copy()
            self.food_rate[idx] = float(food_signal)
            self.risk[idx] = float(risk_signal)
            self.visit_count[idx] = 1.0
        else:
            # EMA update existing node
            idx = best_idx
            alpha = self.ema_rate
            self.food_rate[idx] += alpha * (food_signal - self.food_rate[idx])
            self.risk[idx] += alpha * (risk_signal - self.risk[idx])
            # Centroid drift toward observation
            self.centroids[idx] += self.drift_rate * (z - self.centroids[idx])
            self.visit_count[idx] += 1.0

    def get_saveable_state(self):
        """Return learned memory nodes for checkpoint."""
        return {
            "centroids": self.centroids[:self.n_allocated].copy(),
            "food_rate": self.food_rate[:self.n_allocated].copy(),
            "risk": self.risk[:self.n_allocated].copy(),
            "visit_count": self.visit_count[:self.n_allocated].copy(),
            "n_allocated": self.n_allocated,
        }

    def load_saveable_state(self, state):
        """Restore learned memory nodes."""
        n = state["n_allocated"]
        self.centroids[:] = 0.0
        self.food_rate[:] = 0.0
        self.risk[:] = 0.0
        self.visit_count[:] = 0.0
        self.centroids[:n] = state["centroids"]
        self.food_rate[:n] = state["food_rate"]
        self.risk[:n] = state["risk"]
        self.visit_count[:n] = state["visit_count"]
        self.n_allocated = n

    def reset(self):
        self.centroids[:] = 0.0
        self.food_rate[:] = 0.0
        self.risk[:] = 0.0
        self.visit_count[:] = 0.0
        self.n_allocated = 0


# ---------------------------------------------------------------------------
# VAE World Model (orchestrator)
# ---------------------------------------------------------------------------

class VAEWorldModel:
    """Orchestrates VAE encoding, transition, associative memory, and planning.

    Usage in BrainAgent:
        z, z_mu = vae.encode(oF_tensor, state_ctx)
        vae.train_step(oF_tensor, state_ctx)
        vae.update_transition(z_prev, action_ctx, z)
        G_plan = vae.plan(z, last_action, dopa, cls_probs)
        vae.update_memory(z, eaten, pred_dist)
    """

    def __init__(self, oF_dim=800, pool_dim=64, state_ctx_dim=13,
                 latent_dim=16, act_dim=5,
                 lr=1e-3, kl_beta=0.1, buffer_size=256, batch_size=16,
                 warmup_steps=200, blend_ramp=100, max_blend=0.3,
                 plan_horizon=3, device="cpu"):
        self.oF_dim = oF_dim
        self.pool_dim = pool_dim
        self.state_ctx_dim = state_ctx_dim
        self.latent_dim = latent_dim
        self.kl_beta = kl_beta
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.blend_ramp = blend_ramp
        self.max_blend = max_blend
        self.plan_horizon = plan_horizon
        self.device = device

        # Pool projection: avg_pool 800 → 64 (learned linear for flexibility)
        self.pool = nn.Linear(oF_dim, pool_dim).to(device)
        nn.init.xavier_uniform_(self.pool.weight)
        nn.init.zeros_(self.pool.bias)

        # VAE — encoder takes pooled oF + state context
        enc_input = pool_dim + state_ctx_dim  # 64 + 13 = 77
        self.encoder = VAEEncoder(enc_input, 48, latent_dim).to(device)
        self.decoder = VAEDecoder(latent_dim, 48, pool_dim).to(device)

        # Transition model
        self.transition = TransitionModel(
            latent_dim, act_dim=act_dim, hidden=24).to(device)

        # Associative memory (numpy)
        self.memory = AssociativeMemory(
            n_nodes=64, latent_dim=latent_dim, kernel_width=2.0)

        # Optimizer for VAE + pool (transition trained separately)
        self.vae_params = (
            list(self.pool.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        self.vae_optimizer = torch.optim.Adam(self.vae_params, lr=lr)

        self.trans_optimizer = torch.optim.Adam(
            self.transition.parameters(), lr=lr)

        # Ring buffer for online ELBO training — stores oF + state_ctx
        buf_cols = oF_dim + state_ctx_dim  # 800 + 13 = 813
        self._buffer = np.zeros((buffer_size, buf_cols), dtype=np.float32)
        self._buf_ptr = 0
        self._buf_count = 0

        # Step counter for warmup/blend scheduling
        self._step = 0

        # Diagnostics
        self._last_vae_loss = 0.0
        self._last_trans_loss = 0.0
        self._last_z_mean = np.zeros(latent_dim, dtype=np.float32)
        self._last_G_plan = np.zeros(3, dtype=np.float32)
        self._last_epistemic = np.zeros(3, dtype=np.float32)

    # -------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------

    def encode(self, oF_tensor, state_ctx=None):
        """Encode oF [1, 800] + state_ctx [13] → latent z [16] (numpy).

        Args:
            oF_tensor: torch [1, 800] — fused tectal output
            state_ctx: numpy [13] or None — multi-modal state context.
                If None, zeros are used (backwards compatible).

        Returns:
            z_np: numpy [16] — sampled z (for memory/planning)
            z_mu_np: numpy [16] — mean (for diagnostics)
        """
        with torch.no_grad():
            pooled = F.relu(self.pool(oF_tensor))           # [1, 64]
            # Concatenate state context
            if state_ctx is not None:
                ctx_t = torch.tensor(
                    state_ctx[np.newaxis], dtype=torch.float32,
                    device=self.device)                      # [1, 13]
            else:
                ctx_t = torch.zeros(
                    1, self.state_ctx_dim, device=self.device)
            enc_input = torch.cat([pooled, ctx_t], dim=-1)   # [1, 77]
            mu, logvar = self.encoder(enc_input)              # [1, 16] each
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)              # reparameterize
        z_np = z[0].cpu().numpy().astype(np.float32)
        z_mu_np = mu[0].cpu().numpy().astype(np.float32)
        self._last_z_mean = z_mu_np.copy()
        return z_np, z_mu_np

    # -------------------------------------------------------------------
    # Online ELBO training
    # -------------------------------------------------------------------

    def train_step(self, oF_tensor, state_ctx=None):
        """Push oF + state_ctx to ring buffer and do a mini-batch ELBO update.

        Args:
            oF_tensor: torch [1, 800]
            state_ctx: numpy [13] or None

        Only trains after buffer has enough samples.
        """
        oF_np = oF_tensor[0].detach().cpu().numpy().astype(np.float32)
        ctx_np = (state_ctx if state_ctx is not None
                  else np.zeros(self.state_ctx_dim, dtype=np.float32))
        self._buffer[self._buf_ptr] = np.concatenate([oF_np, ctx_np])
        self._buf_ptr = (self._buf_ptr + 1) % self.buffer_size
        self._buf_count = min(self._buf_count + 1, self.buffer_size)
        self._step += 1

        if self._buf_count < self.batch_size * 2:
            return  # not enough data yet

        # Sample mini-batch from buffer
        indices = np.random.choice(self._buf_count, self.batch_size,
                                   replace=False)
        batch_all = torch.tensor(
            self._buffer[indices], device=self.device)       # [B, 813]
        batch_oF = batch_all[:, :self.oF_dim]                # [B, 800]
        batch_ctx = batch_all[:, self.oF_dim:]               # [B, 13]

        # Forward through pool + encoder + decoder
        pooled = F.relu(self.pool(batch_oF))                 # [B, 64]
        enc_input = torch.cat([pooled, batch_ctx], dim=-1)   # [B, 77]
        mu, logvar = self.encoder(enc_input)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)                              # [B, 64]

        # ELBO loss: reconstruction (pooled oF only) + KL
        recon_loss = F.mse_loss(recon, pooled)
        kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.kl_beta * kl_loss

        self.vae_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae_params, 1.0)
        self.vae_optimizer.step()

        self._last_vae_loss = float(loss.item())

    def train_from_buffer(self):
        """Train ELBO from existing buffer samples only (no push).

        Used during sleep consolidation when sensory input is not meaningful.
        Performs the same mini-batch ELBO update as train_step but without
        pushing new data to the ring buffer.
        """
        if self._buf_count < self.batch_size * 2:
            return

        indices = np.random.choice(self._buf_count, self.batch_size,
                                   replace=False)
        batch_all = torch.tensor(
            self._buffer[indices], device=self.device)
        batch_oF = batch_all[:, :self.oF_dim]
        batch_ctx = batch_all[:, self.oF_dim:]

        pooled = F.relu(self.pool(batch_oF))
        enc_input = torch.cat([pooled, batch_ctx], dim=-1)
        mu, logvar = self.encoder(enc_input)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)

        recon_loss = F.mse_loss(recon, pooled)
        kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.kl_beta * kl_loss

        self.vae_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae_params, 1.0)
        self.vae_optimizer.step()

        self._last_vae_loss = float(loss.item())

    # -------------------------------------------------------------------
    # Transition training
    # -------------------------------------------------------------------

    def update_transition(self, z_prev, action_ctx, z_now):
        """Train transition model from an observed (z, a) → z' pair.

        Args:
            z_prev: numpy [16]
            action_ctx: numpy [5] — [turn, speed, goal_oh[3]]
            z_now: numpy [16]
        """
        z_prev_t = torch.tensor(
            z_prev[np.newaxis], dtype=torch.float32, device=self.device)
        act_t = torch.tensor(
            action_ctx[np.newaxis], dtype=torch.float32, device=self.device)
        z_now_t = torch.tensor(
            z_now[np.newaxis], dtype=torch.float32, device=self.device)

        z_pred = self.transition(z_prev_t, act_t)
        loss = F.mse_loss(z_pred, z_now_t)

        self.trans_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transition.parameters(), 1.0)
        self.trans_optimizer.step()

        self._last_trans_loss = float(loss.item())

    # -------------------------------------------------------------------
    # Associative memory
    # -------------------------------------------------------------------

    def update_memory(self, z, eaten, pred_dist):
        """Update associative memory at latent z with food/risk signals.

        Args:
            z: numpy [16]
            eaten: int — food eaten this step
            pred_dist: float — distance to predator (world units)
        """
        food_signal = float(eaten)
        # Risk inversely proportional to predator distance
        risk_signal = float(max(0.0, 1.0 - pred_dist / 150.0))
        self.memory.update(z, food_signal, risk_signal)

    # -------------------------------------------------------------------
    # Planning
    # -------------------------------------------------------------------

    def get_blend_weight(self):
        """0 during warmup, ramps linearly to max_blend over blend_ramp steps."""
        if self._step < self.warmup_steps:
            return 0.0
        elapsed = self._step - self.warmup_steps
        t = min(1.0, elapsed / max(1, self.blend_ramp))
        return self.max_blend * t

    def _predict_recon_uncertainty(self, z_sim_tensor):
        """Proxy for expected information gain using decoder at simulated z.

        Low decoder output magnitude → unknown region → high uncertainty.
        High output variance → uncertain reconstruction.

        Args:
            z_sim_tensor: torch [1, 16] — simulated latent state

        Returns:
            uncertainty: float [0, 1]
        """
        with torch.no_grad():
            recon = self.decoder(z_sim_tensor)  # [1, 64]
            recon_mag = float(recon.abs().mean())
            recon_var = float(recon.var())
        mag_uncertainty = 1.0 / (1.0 + recon_mag * 5.0)
        var_signal = min(1.0, recon_var * 2.0)
        return 0.5 * mag_uncertainty + 0.5 * var_signal

    def plan(self, z, last_action, dopa, cls_probs):
        """Simulate 3 goals x plan_horizon steps, return G_plan[3].

        Each goal uses a stereotyped action sequence rolled out through
        the transition model. Trajectories are scored by querying
        associative memory for expected food_rate and risk, plus an
        epistemic term capturing expected information gain (curiosity).

        G(π) = pragmatic_value + epistemic_weight * epistemic_value

        Epistemic weights per goal:
          FORAGE:  -0.15 (mild curiosity — prefer novel food patches)
          FLEE:    -0.05 (survival > curiosity)
          EXPLORE: -0.5  (strong epistemic drive — raison d'etre)

        Args:
            z: numpy [16] — current latent state
            last_action: numpy [2] — [turn_rate, speed] from last step
            dopa: float — current dopamine level
            cls_probs: numpy [5] — classifier output

        Returns:
            G_plan: numpy [3] — planning bonus per goal (lower = better)
        """
        blend = self.get_blend_weight()
        if blend < 1e-6:
            self._last_G_plan = np.zeros(3, dtype=np.float32)
            self._last_epistemic = np.zeros(3, dtype=np.float32)
            return self._last_G_plan

        # Stereotyped action sequences per goal
        action_seqs = self._make_action_sequences(last_action)

        # Epistemic weights: how much each goal values information gain
        epistemic_weights = np.array([-0.15, -0.05, -0.5], dtype=np.float32)

        G = np.zeros(3, dtype=np.float32)
        epistemic_vals = np.zeros(3, dtype=np.float32)
        z_t = torch.tensor(
            z[np.newaxis], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for gi in range(3):
                z_sim = z_t.clone()
                total_food = 0.0
                total_risk = 0.0
                z_var_accum = 0.0
                total_novelty = 0.0
                total_model_unc = 0.0

                for step in range(self.plan_horizon):
                    act = torch.tensor(
                        action_seqs[gi][step][np.newaxis],
                        dtype=torch.float32, device=self.device)
                    z_sim = self.transition(z_sim, act)
                    z_np = z_sim[0].cpu().numpy()

                    # Pragmatic signals
                    food_r, risk_r = self.memory.query(z_np)
                    total_food += food_r
                    total_risk += risk_r
                    z_var_accum += float(np.var(z_np))

                    # Epistemic signals
                    novelty, mem_conf = self.memory.query_epistemic(z_np)
                    model_unc = self._predict_recon_uncertainty(z_sim)
                    total_novelty += novelty
                    total_model_unc += model_unc

                H = self.plan_horizon
                avg_novelty = total_novelty / H
                avg_model_unc = total_model_unc / H
                epistemic_val = 0.6 * avg_novelty + 0.4 * avg_model_unc
                epistemic_vals[gi] = epistemic_val

                # Pragmatic score (lower = more attractive for that goal)
                if gi == 0:  # FORAGE
                    pragmatic = -0.6 * total_food + 0.3 * total_risk
                elif gi == 1:  # FLEE
                    pragmatic = 0.8 * total_risk - 0.1 * total_food
                else:  # EXPLORE
                    pragmatic = (-0.3 * total_food
                                 + 0.2 * total_risk
                                 - 0.1 * z_var_accum)

                G[gi] = pragmatic + epistemic_weights[gi] * epistemic_val

        G_plan = G * blend
        self._last_G_plan = G_plan.copy()
        self._last_epistemic = epistemic_vals.copy()
        return G_plan

    def _make_action_sequences(self, last_action):
        """Generate stereotyped action sequences for 3 goals.

        Returns:
            list of 3 lists, each with plan_horizon action_ctx arrays.
            Goal one-hot size matches transition model act_dim.
        """
        turn = last_action[0] if len(last_action) > 0 else 0.0
        speed = last_action[1] if len(last_action) > 1 else 0.5
        H = self.plan_horizon
        # Infer n_goals from transition model act_dim (act_dim = 2 + n_goals)
        n_goals = self.transition.fc1.in_features - self.latent_dim - 2
        seqs = []

        def _goal_oh(idx):
            oh = np.zeros(n_goals, dtype=np.float32)
            oh[idx] = 1.0
            return oh

        # FORAGE: continue forward, moderate speed
        forage_seq = []
        for s in range(H):
            act = np.array([turn * 0.3, min(1.0, speed * 1.1),
                            *_goal_oh(0)], dtype=np.float32)
            forage_seq.append(act)
        seqs.append(forage_seq)

        # FLEE: sharp turn away, accelerate
        flee_seq = []
        flee_dir = 0.8 if turn >= 0 else -0.8
        for s in range(H):
            act = np.array([flee_dir, min(1.0, speed * 1.4),
                            *_goal_oh(1)], dtype=np.float32)
            flee_seq.append(act)
        seqs.append(flee_seq)

        # EXPLORE: gentle weaving, moderate speed
        explore_seq = []
        for s in range(H):
            weave = 0.3 * ((-1) ** s)
            act = np.array([weave, 0.6, *_goal_oh(2)], dtype=np.float32)
            explore_seq.append(act)
        seqs.append(explore_seq)

        return seqs

    # -------------------------------------------------------------------
    # Diagnostics & reset
    # -------------------------------------------------------------------

    def get_diagnostics(self):
        """Return monitoring dict."""
        return {
            "vae_loss": self._last_vae_loss,
            "trans_loss": self._last_trans_loss,
            "z_mean": self._last_z_mean.copy(),
            "blend_weight": self.get_blend_weight(),
            "memory_nodes": self.memory.n_allocated,
            "G_plan": self._last_G_plan.copy(),
            "epistemic_per_goal": self._last_epistemic.copy(),
            "step": self._step,
            "buffer_count": self._buf_count,
        }

    def get_saveable_state(self):
        """Return all learned NN weights, optimizers, and associative memory."""
        return {
            "pool": self.pool.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "transition": self.transition.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict(),
            "trans_optimizer": self.trans_optimizer.state_dict(),
            "memory": self.memory.get_saveable_state(),
            "step": self._step,
        }

    def load_saveable_state(self, state):
        """Restore all learned NN weights, optimizers, and associative memory."""
        self.pool.load_state_dict(state["pool"])
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.transition.load_state_dict(state["transition"])
        self.vae_optimizer.load_state_dict(state["vae_optimizer"])
        self.trans_optimizer.load_state_dict(state["trans_optimizer"])
        self.memory.load_saveable_state(state["memory"])
        self._step = state["step"]

    def reset_episode(self):
        """Clear transient buffers but keep learned weights and memory."""
        self._buffer[:] = 0.0
        self._buf_ptr = 0
        self._buf_count = 0
        self._last_vae_loss = 0.0
        self._last_trans_loss = 0.0
        self._last_z_mean = np.zeros(self.latent_dim, dtype=np.float32)
        self._last_G_plan = np.zeros(3, dtype=np.float32)
        self._last_epistemic = np.zeros(3, dtype=np.float32)

    def reset(self):
        """Clear memory and buffers, keep learned weights."""
        self.memory.reset()
        self._buffer[:] = 0.0
        self._buf_ptr = 0
        self._buf_count = 0
        self._step = 0
        self._last_vae_loss = 0.0
        self._last_trans_loss = 0.0
        self._last_z_mean = np.zeros(self.latent_dim, dtype=np.float32)
        self._last_G_plan = np.zeros(3, dtype=np.float32)
        self._last_epistemic = np.zeros(3, dtype=np.float32)
