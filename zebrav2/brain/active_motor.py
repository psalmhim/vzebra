"""
Active Inference Motor Controller — Friston's action-perception cycle.

Implements the FULL active inference loop (Friston 2011, "What is optimal
about motor control?"):

  1. Generate proprioceptive PREDICTIONS from current goal/beliefs (top-down)
  2. Receive actual proprioception from the body (bottom-up)
  3. Compute PE within two-compartment neurons (apical − somatic)
  4. ITERATIVELY refine predictions to minimize PE (belief update)
  5. Motor output = final PE × reflex gain (classical reflex arc)

The key insight: motor commands ARE proprioceptive predictions.  The body
moves to fulfill those predictions via the spinal reflex arc, which acts
to cancel proprioceptive PE.  This closes the action-perception cycle.

Iterative inference (n_inference_passes):
  Within each step, predictions μ are updated multiple times:
    μ ← μ − η_action × ε   (action = prediction that minimizes PE)
  This allows beliefs to converge before committing to motor output.

Uses TwoCompColumn: each neuron computes PE via apical-somatic mismatch
(Larkum 2013), bias = precision (Lee, Lee & Park 2026).

Architecture:
  TwoCompColumn (8 proprioceptive channels × 6 neurons/channel):
    48 Izhikevich two-compartment neurons
    Apical dendrite: receives top-down motor predictions (goal-dependent)
    Soma: receives bottom-up proprioceptive input
    PE = V_a - norm(V_s): within-neuron, not between populations
    Bias = precision: somatic excitability = gain on PE
    Precision γ: learned via PE magnitude (PrecisionUnit logic)

  Motor output reads from PE and firing rates:
    turn      ← heading channel PE (reflex arc cancelling heading error)
    speed     ← speed channel PE + prediction
    cpg_drive ← tail channel predictions
    cpg_bias  ← tail L/R PE asymmetry
    gaze_pe   ← gaze channel PE

  Precision (bias) modulated by neuromodulators:
    DA  → speed channel (reward approach)
    NA  → heading channel (threat escape)
    ACh → gaze channel (attention)
    5-HT → tail channels (rhythmic/postural)

References:
  Friston (2011) What is optimal about motor control? Neuron
  Larkum (2013) A cellular mechanism for cortical associations. Nature Neuroscience
  Lee, Lee & Park (2026) Modulation of bias in predictive-coding SNNs
  Sacramento, Costa, Bengio & Senn (2018) Dendritic cortical microcircuits. NeurIPS
  Adams, Stephan, Brown, Frith & Friston (2013) Frontiers in Psychiatry
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.two_comp_column import TwoCompColumn

# Goal constants
GOAL_FORAGE = 0
GOAL_FLEE = 1
GOAL_EXPLORE = 2
GOAL_SOCIAL = 3

# Proprioceptive channel indices
CH_HEADING = 0     # heading change (delta theta)
CH_SPEED = 1       # forward velocity
CH_TAIL_L = 2      # left tail muscle activation
CH_TAIL_R = 3      # right tail muscle activation
CH_GAZE = 4        # gaze offset (saccade target)
CH_WALL = 5        # wall proximity (avoidance)
CH_COLLISION = 6   # collision state
CH_TURN_RATE = 7   # angular velocity
N_CHANNELS = 8


class ActiveInferenceMotor(nn.Module):
    """
    Spiking active inference motor controller.

    Motor commands emerge from minimizing PE in two-compartment neurons
    where PE = apical (prediction) − soma (actual proprioception).
    Bias = precision (Lee et al. 2026).
    """

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device

        # --- Two-compartment column (48 Izhikevich neurons) ---
        self.column = TwoCompColumn(
            n_channels=N_CHANNELS,
            n_per_ch=6,
            cell_type='RS',
            substeps=10,
            device=device,
        )

        # EMA smoothing for motor output (motor plant dynamics)
        self._turn_ema = 0.0
        self._speed_ema = 0.5

        # Blend factor: active inference vs reactive motor (adaptive at brain level)
        self.ai_blend = 0.3

        # --- Iterative active inference (Friston 2011) ---
        # Multiple inference passes: predictions adapt to reduce PE before acting.
        # action_lr: how fast predictions (μ) update toward sensory evidence.
        # This IS the action-perception cycle: μ ← μ − η × ε
        self.n_inference_passes = 3
        self.action_lr = 0.15

        # Track per-step inference convergence for diagnostics
        self._fe_per_pass = []

    @torch.no_grad()
    def _compute_prediction_drive(
        self, goal: int, food_bearing: float, enemy_bearing: float,
        wall_proximity: float, food_visible: bool, enemy_visible: bool,
        gaze_target: float, explore_phase: float,
    ) -> torch.Tensor:
        """
        Generate proprioceptive predictions from current beliefs.
        These drive the apical dendrite (top-down prediction).
        """
        mu = torch.zeros(N_CHANNELS, device=self.device)

        if goal == GOAL_FLEE:
            mu[CH_HEADING] = -enemy_bearing * 0.8
            mu[CH_SPEED] = 0.9
            mu[CH_TAIL_L] = 0.6
            mu[CH_TAIL_R] = 0.6
            if enemy_bearing > 0:
                mu[CH_TAIL_R] += 0.2
            else:
                mu[CH_TAIL_L] += 0.2
            mu[CH_GAZE] = enemy_bearing * 0.2

        elif goal == GOAL_FORAGE:
            if food_visible:
                mu[CH_HEADING] = food_bearing * 0.5
                mu[CH_SPEED] = 0.4
                mu[CH_TAIL_L] = 0.3
                mu[CH_TAIL_R] = 0.3
                if food_bearing > 0.1:
                    mu[CH_TAIL_R] += 0.1
                elif food_bearing < -0.1:
                    mu[CH_TAIL_L] += 0.1
                mu[CH_GAZE] = food_bearing * 0.3
            else:
                mu[CH_SPEED] = 0.3
                mu[CH_TAIL_L] = 0.2
                mu[CH_TAIL_R] = 0.2
                mu[CH_GAZE] = explore_phase * 0.3

        elif goal == GOAL_EXPLORE:
            mu[CH_HEADING] = explore_phase * 0.3
            mu[CH_SPEED] = 0.3
            mu[CH_TAIL_L] = 0.2
            mu[CH_TAIL_R] = 0.2
            mu[CH_GAZE] = explore_phase * 0.4

        else:  # SOCIAL
            mu[CH_SPEED] = 0.25
            mu[CH_TAIL_L] = 0.15
            mu[CH_TAIL_R] = 0.15

        mu[CH_WALL] = 0.0
        mu[CH_COLLISION] = 0.0
        mu[CH_TURN_RATE] = mu[CH_HEADING]

        return mu

    @torch.no_grad()
    def _compute_sensory_drive(
        self, actual_speed: float, heading_delta: float,
        tail_L: float, tail_R: float, gaze_offset: float,
        wall_proximity: float, collision: bool, turn_rate: float,
    ) -> torch.Tensor:
        """Encode actual proprioceptive state (drives soma)."""
        y = torch.zeros(N_CHANNELS, device=self.device)
        y[CH_HEADING] = heading_delta
        y[CH_SPEED] = min(1.0, actual_speed / 4.0)
        y[CH_TAIL_L] = tail_L
        y[CH_TAIL_R] = tail_R
        y[CH_GAZE] = gaze_offset
        y[CH_WALL] = wall_proximity
        y[CH_COLLISION] = 1.0 if collision else 0.0
        y[CH_TURN_RATE] = turn_rate
        return y

    @torch.no_grad()
    def _modulate_precision(self, DA: float, NA: float,
                            HT5: float, ACh: float):
        """
        Neuromodulatory control of precision via attention signal.
        In the two-compartment model, precision = bias = somatic excitability.
        """
        att = torch.zeros(N_CHANNELS, device=self.device)
        att[CH_SPEED] = DA * 3.0         # DA → speed precision
        att[CH_HEADING] = NA * 5.0       # NA → heading precision (threat)
        att[CH_GAZE] = ACh * 4.0         # ACh → gaze precision (attention)
        att[CH_TAIL_L] = HT5 * 2.5      # 5-HT → tail precision (rhythmic)
        att[CH_TAIL_R] = HT5 * 2.5
        att[CH_WALL] = NA * 2.0          # NA also helps wall avoidance
        att[CH_COLLISION] = NA * 3.0     # collision is salient under threat
        self.column.set_attention(att)

    @torch.no_grad()
    def step(self, goal: int,
             food_bearing: float, enemy_bearing: float,
             wall_proximity: float, food_visible: bool, enemy_visible: bool,
             gaze_target: float, explore_phase: float,
             DA: float, NA: float, HT5: float, ACh: float,
             actual_speed: float, heading_delta: float,
             tail_L: float, tail_R: float,
             gaze_offset: float, collision: bool, turn_rate: float,
             ) -> dict:
        """
        Full action-perception cycle (Friston 2011).

        Iterates multiple inference passes per step:
          Pass 1: initial prediction → PE → belief update
          Pass 2: refined prediction → smaller PE → belief update
          Pass 3: converged prediction → final PE → motor output

        Motor output = reflex arc reading PE from final pass.
        Action = proprioceptive prediction the body fulfills.
        """
        # 1. Set precision via neuromodulators (adjusts bias)
        self._modulate_precision(DA, NA, HT5, ACh)

        # 2. Top-down predictions → apical dendrite (initial beliefs)
        mu = self._compute_prediction_drive(
            goal, food_bearing, enemy_bearing, wall_proximity,
            food_visible, enemy_visible, gaze_target, explore_phase)

        # 3. Bottom-up proprioception → soma (fixed within step)
        y = self._compute_sensory_drive(
            actual_speed, heading_delta, tail_L, tail_R,
            gaze_offset, wall_proximity, collision, turn_rate)

        # 4. Iterative active inference: predictions converge toward
        #    sensory evidence through PE-driven belief updates.
        #    μ ← μ − η × ε  (Friston 2011, eq. 4)
        #    This IS the action-perception cycle: each pass reduces PE
        #    by adjusting what the motor system predicts/commands.
        self._fe_per_pass = []
        for pass_i in range(self.n_inference_passes):
            col_out = self.column(sensory_drive=y, prediction_drive=mu)
            pe = col_out['pe']
            self._fe_per_pass.append(col_out['free_energy'])

            # Update predictions to reduce PE (action as prediction)
            # On last pass, keep PE for motor readout — don't update mu
            if pass_i < self.n_inference_passes - 1:
                mu = mu - self.action_lr * pe
                mu = mu.clamp(-1.5, 1.5)

        pe = col_out['pe']       # final-pass PE
        rate = col_out['rate']

        # 5. Motor output: reflex arc reading PE (PE drives the body
        #    to fulfill the prediction — classical motor reflex arc)
        pe_heading = float(pe[CH_HEADING])
        pe_turn = float(pe[CH_TURN_RATE])
        ai_turn = 0.6 * pe_heading + 0.4 * pe_turn
        ai_turn = max(-1.0, min(1.0, ai_turn * 3.0))

        pe_speed = float(pe[CH_SPEED])
        ai_speed = float(mu[CH_SPEED]) + pe_speed * 1.5
        ai_speed = max(0.0, min(1.5, ai_speed))

        # CPG from converged predictions and PE
        ai_cpg_drive = (float(mu[CH_TAIL_L]) + float(mu[CH_TAIL_R])) / 2.0
        pe_tail_L = float(pe[CH_TAIL_L])
        pe_tail_R = float(pe[CH_TAIL_R])
        ai_cpg_bias = (pe_tail_R - pe_tail_L) * 2.0

        pe_gaze = float(pe[CH_GAZE])

        # Wall/collision PE → avoidance
        pe_wall = float(pe[CH_WALL])
        pe_collision = float(pe[CH_COLLISION])
        wall_correction = -(pe_wall + pe_collision * 2.0) * 1.5
        ai_turn = max(-1.0, min(1.0, ai_turn + wall_correction))

        # 6. EMA smoothing (motor plant dynamics)
        alpha_turn = 0.4
        alpha_speed = 0.3
        self._turn_ema = (1 - alpha_turn) * self._turn_ema + alpha_turn * ai_turn
        self._speed_ema = (1 - alpha_speed) * self._speed_ema + alpha_speed * ai_speed

        # Inference convergence: how much FE decreased across passes
        _fe_converged = (self._fe_per_pass[0] - self._fe_per_pass[-1]
                         if len(self._fe_per_pass) > 1 else 0.0)

        return {
            'turn': float(self._turn_ema),
            'speed': float(self._speed_ema),
            'cpg_drive': max(0.0, min(1.0, ai_cpg_drive)),
            'cpg_bias': max(-1.0, min(1.0, ai_cpg_bias)),
            'gaze_pe': pe_gaze,
            'free_energy': col_out['free_energy'],
            'precision': col_out['precision'].detach().cpu().tolist(),
            'prediction_error': pe.detach().cpu().tolist(),
            'pred_rate': float(rate.mean()),
            'sens_rate': float(col_out['spikes_mean']),
            'error_rate': float(pe.abs().mean()),
            'inference_convergence': _fe_converged,
            'n_passes': self.n_inference_passes,
        }

    def get_precision_profile(self) -> dict:
        """Named precision values for diagnostics."""
        pi = self.column.precision
        return {
            'heading': float(pi[CH_HEADING]),
            'speed': float(pi[CH_SPEED]),
            'tail_L': float(pi[CH_TAIL_L]),
            'tail_R': float(pi[CH_TAIL_R]),
            'gaze': float(pi[CH_GAZE]),
            'wall': float(pi[CH_WALL]),
            'collision': float(pi[CH_COLLISION]),
            'turn_rate': float(pi[CH_TURN_RATE]),
        }

    def reset(self):
        self.column.reset()
        self._turn_ema = 0.0
        self._speed_ema = 0.5
