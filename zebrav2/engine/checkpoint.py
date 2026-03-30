"""
Checkpoint manager: save/load all learnable brain state.
"""
import os
import json
import time
import torch
import numpy as np


class CheckpointManager:
    def __init__(self, save_dir='zebrav2/checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, brain, round_num, metrics, config=None):
        """Save all learnable state from brain."""
        ckpt = {
            'round': round_num,
            'timestamp': time.time(),
            'metrics': metrics,
            # RL Critic
            'critic_state': brain.critic.state_dict(),
            # Classifier
            'classifier_state': brain.classifier.state_dict(),
            # Pallium (includes W_FB)
            'pallium_state': brain.pallium.state_dict(),
            # Cerebellum (parallel fiber weights)
            'cerebellum_W_pf': brain.cerebellum.W_pf.data.cpu().numpy().tolist(),
            # Amygdala (LA→CeA weights + fear baseline)
            'amygdala_W_la_cea': brain.amygdala.W_la_cea.cpu().numpy().tolist(),
            'amygdala_fear_baseline': brain.amygdala.fear_baseline,
            # Habit network
            'habit_state': brain.habit.state_dict(),
            # Place cells
            'place_food_rate': brain.place.food_rate.cpu().numpy().tolist(),
            'place_risk_rate': brain.place.risk_rate.cpu().numpy().tolist(),
            'place_visit_count': brain.place.visit_count.cpu().numpy().tolist(),
            # Geographic model
            'geo_food': brain.geo_model.food_score.tolist(),
            'geo_risk': brain.geo_model.risk_score.tolist(),
            'geo_visits': brain.geo_model.visit_count.tolist(),
            # VAE world model
            'vae_pool': brain.vae.pool.state_dict(),
            'vae_encoder': brain.vae.encoder.state_dict(),
            'vae_decoder': brain.vae.decoder.state_dict(),
            'vae_transition': brain.vae.transition.state_dict(),
            'vae_memory_centroids': brain.vae.memory.centroids[:brain.vae.memory.n_allocated].tolist(),
            'vae_memory_food': brain.vae.memory.food_rate[:brain.vae.memory.n_allocated].tolist(),
            'vae_memory_risk': brain.vae.memory.risk[:brain.vae.memory.n_allocated].tolist(),
            'vae_memory_n': brain.vae.memory.n_allocated,
            # Habenula frustration
            'habenula_frustration': brain.habenula.frustration.tolist(),
            # Personality
            'personality': brain.personality,
            # Neuromod baselines
            'neuromod_DA': brain.neuromod.DA.item(),
            'neuromod_V': brain.neuromod.V.item(),
        }
        if config:
            ckpt['config'] = config.data if hasattr(config, 'data') else config

        path = os.path.join(self.save_dir, f'ckpt_round_{round_num:04d}.pt')
        torch.save(ckpt, path)
        # Save metadata as JSON for web dashboard
        meta = {
            'round': round_num,
            'timestamp': ckpt['timestamp'],
            'metrics': metrics,
            'path': path,
        }
        meta_path = os.path.join(self.save_dir, f'ckpt_round_{round_num:04d}.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        return path

    def load(self, brain, path):
        """Load checkpoint into brain."""
        ckpt = torch.load(path, map_location=brain.device, weights_only=False)

        brain.critic.load_state_dict(ckpt['critic_state'])
        brain.classifier.load_state_dict(ckpt['classifier_state'])
        brain.pallium.load_state_dict(ckpt['pallium_state'])
        brain.habit.load_state_dict(ckpt['habit_state'])

        brain.cerebellum.W_pf.data = torch.tensor(
            ckpt['cerebellum_W_pf'], device=brain.device, dtype=torch.float32)
        brain.amygdala.W_la_cea.copy_(torch.tensor(
            ckpt['amygdala_W_la_cea'], device=brain.device, dtype=torch.float32))
        brain.amygdala.fear_baseline = ckpt['amygdala_fear_baseline']

        brain.place.food_rate.copy_(torch.tensor(
            ckpt['place_food_rate'], device=brain.device, dtype=torch.float32))
        brain.place.risk_rate.copy_(torch.tensor(
            ckpt['place_risk_rate'], device=brain.device, dtype=torch.float32))
        brain.place.visit_count.copy_(torch.tensor(
            ckpt['place_visit_count'], device=brain.device, dtype=torch.float32))

        brain.geo_model.food_score = np.array(ckpt['geo_food'], dtype=np.float32)
        brain.geo_model.risk_score = np.array(ckpt['geo_risk'], dtype=np.float32)
        brain.geo_model.visit_count = np.array(ckpt['geo_visits'], dtype=np.float32)

        brain.vae.pool.load_state_dict(ckpt['vae_pool'])
        brain.vae.encoder.load_state_dict(ckpt['vae_encoder'])
        brain.vae.decoder.load_state_dict(ckpt['vae_decoder'])
        brain.vae.transition.load_state_dict(ckpt['vae_transition'])
        n = ckpt['vae_memory_n']
        brain.vae.memory.centroids[:n] = np.array(ckpt['vae_memory_centroids'])
        brain.vae.memory.food_rate[:n] = np.array(ckpt['vae_memory_food'])
        brain.vae.memory.risk[:n] = np.array(ckpt['vae_memory_risk'])
        brain.vae.memory.n_allocated = n

        brain.habenula.frustration = np.array(ckpt['habenula_frustration'], dtype=np.float32)
        if 'personality' in ckpt:
            brain.personality = ckpt['personality']

        return ckpt.get('round', 0), ckpt.get('metrics', {})

    def list_checkpoints(self):
        """List all available checkpoints."""
        ckpts = []
        for f in sorted(os.listdir(self.save_dir)):
            if f.endswith('.json'):
                with open(os.path.join(self.save_dir, f)) as fp:
                    ckpts.append(json.load(fp))
        return ckpts
