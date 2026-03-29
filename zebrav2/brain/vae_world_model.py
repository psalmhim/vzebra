"""
VAE World Model for active inference planning.

Encodes tectal output + state context into latent z, learns transition
model for action-conditioned prediction, uses associative memory for
planning bonuses.

Components:
  - VAEEncoder/Decoder: [pooled tectum (64) + state ctx (13)] → z (16) → recon (64)
  - TransitionModel: z' = z + 0.3 * delta(z, action_ctx)
  - AssociativeMemory: 64 RBF nodes mapping z → (food_rate, risk)
  - Online ELBO training via ring buffer

Ported from v1 with v2 device handling.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, in_dim=77, hidden=48, latent=16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent=16, hidden=48, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc_out(h)


class TransitionModel(nn.Module):
    RESIDUAL_SCALE = 0.3

    def __init__(self, latent=16, act_dim=5, hidden=24):
        super().__init__()
        self.fc1 = nn.Linear(latent + act_dim, hidden)
        self.fc_out = nn.Linear(hidden, latent)

    def forward(self, z, action_ctx):
        h = F.relu(self.fc1(torch.cat([z, action_ctx], dim=-1)))
        delta = torch.tanh(self.fc_out(h))
        return z + self.RESIDUAL_SCALE * delta


class AssociativeMemory:
    """RBF-based memory: z → (food_rate, risk)."""
    def __init__(self, n_nodes=64, latent_dim=16, kernel_width=2.0,
                 match_threshold=3.0, ema_rate=0.1):
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.kernel_width = kernel_width
        self.match_threshold = match_threshold
        self.ema_rate = ema_rate
        self.centroids = np.zeros((n_nodes, latent_dim), dtype=np.float32)
        self.food_rate = np.zeros(n_nodes, dtype=np.float32)
        self.risk = np.zeros(n_nodes, dtype=np.float32)
        self.visit_count = np.zeros(n_nodes, dtype=np.float32)
        self.n_allocated = 0

    def _rbf_weights(self, z):
        if self.n_allocated == 0:
            return np.zeros(self.n_nodes, dtype=np.float32)
        active = self.centroids[:self.n_allocated]
        dist_sq = np.sum((active - z[np.newaxis, :]) ** 2, axis=1)
        w = np.exp(-dist_sq / (2.0 * self.kernel_width ** 2))
        full_w = np.zeros(self.n_nodes, dtype=np.float32)
        full_w[:self.n_allocated] = w
        return full_w

    def query(self, z):
        w = self._rbf_weights(z)
        w_sum = w.sum() + 1e-8
        return float(np.dot(w, self.food_rate) / w_sum), float(np.dot(w, self.risk) / w_sum)

    def query_epistemic(self, z):
        if self.n_allocated == 0:
            return 1.0, 0.0
        w = self._rbf_weights(z)
        w_sum = w.sum() + 1e-8
        raw = float(np.dot(w, self.visit_count) / w_sum)
        novelty = 1.0 - raw / (raw + 5.0)
        max_w = float(w[:self.n_allocated].max())
        return novelty, max_w

    def update(self, z, food_signal, risk_signal):
        w = self._rbf_weights(z)
        best_idx = int(np.argmax(w[:max(self.n_allocated, 1)]))
        best_w = w[best_idx] if self.n_allocated > 0 else 0.0
        threshold = np.exp(-self.match_threshold ** 2 / 2.0)
        if best_w < threshold:
            if self.n_allocated < self.n_nodes:
                idx = self.n_allocated
                self.n_allocated += 1
            else:
                idx = int(np.argmin(self.visit_count))
            self.centroids[idx] = z.copy()
            self.food_rate[idx] = float(food_signal)
            self.risk[idx] = float(risk_signal)
            self.visit_count[idx] = 1.0
        else:
            idx = best_idx
            alpha = self.ema_rate
            self.food_rate[idx] += alpha * (food_signal - self.food_rate[idx])
            self.risk[idx] += alpha * (risk_signal - self.risk[idx])
            self.centroids[idx] += 0.02 * (z - self.centroids[idx])
            self.visit_count[idx] += 1.0

    def reset(self):
        self.centroids[:] = 0
        self.food_rate[:] = 0
        self.risk[:] = 0
        self.visit_count[:] = 0
        self.n_allocated = 0


class VAEWorldModelV2:
    """VAE world model with online ELBO training and planning."""

    def __init__(self, tectum_dim=800, pool_dim=64, state_ctx_dim=13,
                 latent_dim=16, act_dim=5, lr=1e-3, kl_beta=0.1,
                 buffer_size=256, batch_size=16, warmup_steps=200,
                 max_blend=0.3, plan_horizon=3, device='cpu'):
        self.pool_dim = pool_dim
        self.state_ctx_dim = state_ctx_dim
        self.latent_dim = latent_dim
        self.tectum_dim = tectum_dim
        self.kl_beta = kl_beta
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.max_blend = max_blend
        self.plan_horizon = plan_horizon
        self.device = device

        # Pool projection
        self.pool = nn.Linear(tectum_dim, pool_dim).to(device)
        nn.init.xavier_uniform_(self.pool.weight)

        enc_input = pool_dim + state_ctx_dim  # 77
        self.encoder = VAEEncoder(enc_input, 48, latent_dim).to(device)
        self.decoder = VAEDecoder(latent_dim, 48, pool_dim).to(device)
        self.transition = TransitionModel(latent_dim, act_dim, 24).to(device)
        self.memory = AssociativeMemory(64, latent_dim)

        self.vae_params = (list(self.pool.parameters())
                           + list(self.encoder.parameters())
                           + list(self.decoder.parameters()))
        self.vae_optimizer = torch.optim.Adam(self.vae_params, lr=lr)
        self.trans_optimizer = torch.optim.Adam(self.transition.parameters(), lr=lr)

        buf_cols = tectum_dim + state_ctx_dim
        self._buffer = np.zeros((buffer_size, buf_cols), dtype=np.float32)
        self._buf_ptr = 0
        self._buf_count = 0
        self._step = 0
        self._last_vae_loss = 0.0
        self._last_trans_loss = 0.0
        self._last_z = np.zeros(latent_dim, dtype=np.float32)
        self._last_G_plan = np.zeros(3, dtype=np.float32)

    def encode(self, tectum_tensor, state_ctx=None):
        """Encode tectum [1, N] + state_ctx [13] → z [16]."""
        with torch.no_grad():
            pooled = F.relu(self.pool(tectum_tensor))
            if state_ctx is not None:
                ctx_t = torch.tensor(state_ctx[np.newaxis], dtype=torch.float32, device=self.device)
            else:
                ctx_t = torch.zeros(1, self.state_ctx_dim, device=self.device)
            enc_input = torch.cat([pooled, ctx_t], dim=-1)
            mu, logvar = self.encoder(enc_input)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        z_np = z[0].cpu().numpy().astype(np.float32)
        self._last_z = z_np.copy()
        return z_np, mu[0].cpu().numpy().astype(np.float32)

    def train_step(self, tectum_tensor, state_ctx=None):
        """Push to buffer and do mini-batch ELBO update."""
        oF_np = tectum_tensor[0].detach().cpu().numpy().astype(np.float32)
        ctx_np = state_ctx if state_ctx is not None else np.zeros(self.state_ctx_dim, dtype=np.float32)
        self._buffer[self._buf_ptr] = np.concatenate([oF_np, ctx_np])
        self._buf_ptr = (self._buf_ptr + 1) % len(self._buffer)
        self._buf_count = min(self._buf_count + 1, len(self._buffer))
        self._step += 1
        if self._buf_count < self.batch_size * 2:
            return
        indices = np.random.choice(self._buf_count, self.batch_size, replace=False)
        batch_all = torch.tensor(self._buffer[indices], device=self.device)
        batch_oF = batch_all[:, :self.tectum_dim]
        batch_ctx = batch_all[:, self.tectum_dim:]
        pooled = F.relu(self.pool(batch_oF))
        enc_input = torch.cat([pooled, batch_ctx], dim=-1)
        mu, logvar = self.encoder(enc_input)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, pooled)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.kl_beta * kl_loss
        self.vae_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae_params, 1.0)
        self.vae_optimizer.step()
        self._last_vae_loss = float(loss.item())

    def update_transition(self, z_prev, action_ctx, z_now):
        """Train transition from (z, a) → z'."""
        z_prev_t = torch.tensor(z_prev[np.newaxis], dtype=torch.float32, device=self.device)
        act_t = torch.tensor(action_ctx[np.newaxis], dtype=torch.float32, device=self.device)
        z_now_t = torch.tensor(z_now[np.newaxis], dtype=torch.float32, device=self.device)
        z_pred = self.transition(z_prev_t, act_t)
        loss = F.mse_loss(z_pred, z_now_t)
        self.trans_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transition.parameters(), 1.0)
        self.trans_optimizer.step()
        self._last_trans_loss = float(loss.item())

    def update_memory(self, z, eaten, pred_dist):
        food_signal = float(eaten)
        risk_signal = float(max(0.0, 1.0 - pred_dist / 150.0))
        self.memory.update(z, food_signal, risk_signal)

    def plan(self, z, last_action, dopa=0.5):
        """Simulate 3 goals x horizon, return G_plan[3]."""
        blend = min(self.max_blend, max(0.0, (self._step - self.warmup_steps) / 100.0) * self.max_blend)
        if blend < 1e-6:
            self._last_G_plan = np.zeros(3, dtype=np.float32)
            return self._last_G_plan
        turn = last_action[0] if len(last_action) > 0 else 0.0
        speed = last_action[1] if len(last_action) > 1 else 0.5
        H = self.plan_horizon
        # Stereotyped action sequences
        seqs = []
        for gi in range(3):
            seq = []
            for s in range(H):
                if gi == 0:  # FORAGE
                    act = np.array([turn * 0.3, min(1.0, speed * 1.1), 1, 0, 0], dtype=np.float32)
                elif gi == 1:  # FLEE
                    act = np.array([0.8 if turn >= 0 else -0.8, min(1.0, speed * 1.4), 0, 1, 0], dtype=np.float32)
                else:  # EXPLORE
                    act = np.array([0.3 * ((-1) ** s), 0.6, 0, 0, 1], dtype=np.float32)
                seq.append(act)
            seqs.append(seq)
        G = np.zeros(3, dtype=np.float32)
        z_t = torch.tensor(z[np.newaxis], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for gi in range(3):
                z_sim = z_t.clone()
                total_food, total_risk = 0.0, 0.0
                for s in range(H):
                    act = torch.tensor(seqs[gi][s][np.newaxis], dtype=torch.float32, device=self.device)
                    z_sim = self.transition(z_sim, act)
                    z_np = z_sim[0].cpu().numpy()
                    food_r, risk_r = self.memory.query(z_np)
                    total_food += food_r
                    total_risk += risk_r
                    novelty, _ = self.memory.query_epistemic(z_np)
                if gi == 0:
                    G[gi] = -0.6 * total_food + 0.3 * total_risk
                elif gi == 1:
                    G[gi] = 0.8 * total_risk - 0.1 * total_food
                else:
                    G[gi] = -0.3 * total_food + 0.2 * total_risk - 0.5 * novelty
        G_plan = G * blend
        self._last_G_plan = G_plan.copy()
        return G_plan

    def get_diagnostics(self):
        return {
            'vae_loss': self._last_vae_loss,
            'trans_loss': self._last_trans_loss,
            'memory_nodes': self.memory.n_allocated,
            'G_plan': self._last_G_plan.copy(),
            'step': self._step,
            'z': self._last_z.copy(),
        }

    def reset(self):
        self.memory.reset()
        self._buffer[:] = 0
        self._buf_ptr = 0
        self._buf_count = 0
        self._step = 0
        self._last_vae_loss = 0.0
        self._last_trans_loss = 0.0
        self._last_z[:] = 0
        self._last_G_plan[:] = 0
