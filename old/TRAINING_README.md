# Zebrafish Active Inference Training Pipeline

## Overview

This training pipeline implements a neurobiologically inspired zebrafish agent that learns to distinguish prey from predators and adapt its behavior through active inference.

## Architecture

### 1. Sensory Processing
- **RetinaRenderer**: Converts object position/size to ON/OFF retinal channels
- **RetinaPC**: Predictive coding on retinal input → computes visual free energy
- **VisualCortexPC (V1)**: Cortical encoding → latent features

### 2. Memory Systems
- **WorkingMemory**: Fast, dopamine-modulated short-term memory
- **CortexMemory**: Slower cortical integration

### 3. Dopamine Systems
- **DopamineTectalSystem**: Fast, sensory-driven dopamine (surprise-based)
- **DopamineHierarchicalEFESystem**: Slow, policy-level dopamine (EFE-based)

### 4. Policy & Motor
- **GoalPolicyField**: Active inference policy (approach/flee/neutral)
- **MotorTail**: Tail movement patterns
- **OpticTectum**: Gaze control

### 5. Classifier
- **prey_pred**: Linear classifier on V1 latents (trained supervised)

## Training Stages

### Stage 1: Dataset Generation
```python
X, y = build_dataset(N=2000, device=device)
```
- Samples N prey/predator objects from environment
- Renders retinal inputs
- Extracts V1 latent features
- Returns: `X` [N, 64], `y` [N] (0=prey, 1=predator)

### Stage 2: Supervised Classifier Training
```python
agent.train_classifier(X, y, epochs=20, lr=1e-3)
```
- Trains linear classifier to distinguish prey vs predator
- Uses CrossEntropyLoss
- Reports per-epoch loss and accuracy
- Final output: overall, prey, and predator recognition rates

### Stage 3: RL Policy Fine-tuning
```python
for t in range(1500):
    out = agent.step(retL, retR)
    reward = agent.rl_update_policy(prey_prob, pred_prob, choice, label)
```
- Tests classifier in closed-loop with policy
- Updates policy based on behavioral success
- Correct actions: prey→approach, predator→flee
- Tracks running accuracy

## Usage

### Training from scratch
```bash
python train_agent.py
```

Expected output:
```
STAGE 1: Building supervised dataset
  Generated 2000/2000 samples...
  ✓ Dataset built: X shape: [2000, 64]

STAGE 2: Training prey/predator classifier
  [Classifier] Epoch  1/20  Loss=0.6931  Acc=50.0%
  [Classifier] Epoch  6/20  Loss=0.4312  Acc=78.5%
  ...
  ✓ Supervised classifier training complete
    Overall accuracy: 95.2%
    Prey recognition: 94.8%
    Predator recognition: 95.6%

STAGE 3: RL fine-tuning of policy
  [RL] Step    0  Reward=+1.0  Running acc=62.3%
  ...
  ✓ RL fine-tuning complete
    Final accuracy: 89.7%
```

Outputs:
- `plots/rl_rewards.png` - RL training curve
- `models/prey_pred.pt` - Trained classifier weights

### Evaluating trained model
```bash
python evaluate_classifier.py
```

Outputs:
- Test set accuracy metrics
- Confusion matrix
- Confidence distributions
- `plots/classifier_evaluation.png`

### Running simulation
```bash
python zebra_simulation.py
```

Outputs:
- Real-time agent behavior
- Visual FE, dopamine, predictions, policy over time
- `plots/simulation_summary.png`

## File Structure

```
train_agent.py              # Main training pipeline
dataset_builder.py          # Dataset generation
evaluate_classifier.py      # Classifier evaluation
zebra_simulation.py         # Closed-loop simulation
zebrafish_agent.py          # Full agent implementation
prey_predator_env.py        # Environment simulator

modules/
  retina_renderer.py        # Visual encoding
  retina_pc.py              # Retinal predictive coding
  visual_cortex_pc.py       # V1 cortex
  working_memory.py         # Short-term memory
  dopamine_*.py             # DA systems
  policy_field.py           # Policy selection
  motor_tail.py             # Motor patterns
  optic_tectum.py           # Gaze control
  ...

plots/                      # Training & evaluation plots
models/                     # Saved model weights
```

## Key Parameters

### Dataset
- `N=2000`: Number of training samples
- Labels: 0=prey, 1=predator (neutral excluded)

### Classifier
- `epochs=20`: Training epochs
- `lr=1e-3`: Learning rate
- Architecture: Linear(64 → 2)

### RL
- `steps=1500`: Policy fine-tuning steps
- Reward: +1.0 (correct), -1.0 (incorrect)
- Actions: 0=approach, 1=flee, 2=neutral

## Expected Performance

After training:
- **Classifier accuracy**: 90-95%
- **Prey recognition**: 90-95%
- **Predator recognition**: 90-95%
- **RL accuracy**: 85-90% (includes policy uncertainty)

Random baseline: 50%

## Troubleshooting

### Low accuracy
- Increase dataset size: `N=5000`
- More epochs: `epochs=50`
- Check label distribution (should be ~50/50)

### Poor RL performance
- Classifier may not be confident enough
- Try more supervised training first
- Check dopamine signals in simulation

### Shape errors
- Ensure `y` is 1D integer tensor: `y.shape = [N]`, `dtype=torch.long`
- X should be 2D: `X.shape = [N, 64]`

## References

- Active Inference: Friston et al.
- Zebrafish neuroscience: Portugues & Engert
- Predictive coding: Rao & Ballard
