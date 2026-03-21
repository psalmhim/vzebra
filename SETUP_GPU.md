# GPU Migration Guide — vzebra

## Quick Setup (any Linux GPU machine)

```bash
# Clone
git clone https://github.com/psalmhim/vzebra.git
cd vzebra

# Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run training pipeline (in order)
python -m zebrav1.tests.step8_genomic_pretraining
python -m zebrav1.tests.step10_hebbian_finetuning
python -m zebrav1.tests.step11_object_classification
python -m zebrav1.tests.step11b_gameplay_classifier
python -m zebrav1.tests.step26_wfb_pe_training

# Online RL (100+ episodes)
python -m zebrav1.tests.step43_online_rl

# Spiking distillation + RL
python -m zebrav1.tests.step44_spiking_distillation

# Evaluation
python -m zebrav1.tests.step29b_decision_scenarios
python -m zebrav1.tests.step30_curriculum_motor

# Dashboard (no display needed — web-based)
python -m zebrav1.dashboard
# Open http://<server-ip>:8000

# Demo (headless recording — no display needed)
python -m zebrav1.gym_env.demo --brain --record --monitor --steps 1000
```

## Cloud Providers

### Lambda Labs (recommended: A100)
```bash
# $1.10/hr, A100 80GB
ssh ubuntu@<ip>
sudo apt update && sudo apt install -y python3-venv python3-pip
git clone https://github.com/psalmhim/vzebra.git && cd vzebra
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Google Colab (free T4)
```python
!git clone https://github.com/psalmhim/vzebra.git
%cd vzebra
!pip install -r requirements.txt
!python -m zebrav1.tests.step43_online_rl
```

### RunPod (H100)
```bash
# Use PyTorch template, then:
git clone https://github.com/psalmhim/vzebra.git && cd vzebra
pip install -r requirements.txt
```

## Device Auto-Detection

The code automatically selects the best device:
- `brain/device_util.py`: `get_device()` returns "cuda" > "mps" > "cpu"
- No code changes needed for GPU migration

## Weight Files (not in git — must train or download)

Training produces these in `zebrav1/weights/`:
- `genomic.pt` — Step 8 (5 min on A100)
- `genomic_hebbian.pt` — Step 10 (3 min)
- `classifier.pt` — Step 11 (2 hrs on MPS, ~20 min on A100)
- `classifier_wfb.pt` — Step 26 (5 min)
- `brain_checkpoint.pt` — RL checkpoint (ongoing)
- `spiking_distilled.pt` — Spiking module weights
- `gameplay_classifier_data.npz` — Training data

## Expected Training Times

| Task | MPS (Mac) | T4 (Colab) | A100 | H100 |
|------|-----------|------------|------|------|
| Full pipeline (step8→26) | 2 hrs | 40 min | 20 min | 10 min |
| 100-ep RL | 2.3 hrs | 45 min | 20 min | 9 min |
| 1000-ep RL | 23 hrs | 5 hrs | 3.3 hrs | 1.5 hrs |
| Spiking distillation | 45 min | 15 min | 7 min | 3 min |

## Project Structure

```
vzebra/
├── CLAUDE.md              # AI assistant instructions
├── README.md              # Project overview
├── manuscript.tex         # Neural Computation paper (32 pages)
├── requirements.txt       # Python dependencies
├── SETUP_GPU.md          # This file
├── zebrav1/
│   ├── brain/            # 36+ neural modules
│   ├── gym_env/          # Gymnasium environment + BrainAgent
│   ├── world/            # Ray-casting world renderer
│   ├── viz/              # Neural monitor + sound engine
│   ├── dashboard/        # Web dashboard (FastAPI)
│   ├── tests/            # Training + evaluation scripts
│   ├── weights/          # Trained weights (git-ignored)
│   └── paper.tex         # Technical report (68 pages)
└── plots/                # Generated figures
```

## Key Commands

```bash
# Training
python -m zebrav1.tests.step8_genomic_pretraining      # SNN pretraining
python -m zebrav1.tests.step43_online_rl                # RL training
python -m zebrav1.tests.step44_spiking_distillation     # Spiking training

# Evaluation
python -m zebrav1.tests.step29b_decision_scenarios      # 84/100 target
python -m zebrav1.tests.step30_curriculum_motor          # 4/4 target

# Demo (headless)
python -m zebrav1.gym_env.demo --brain --record --monitor --steps 1000

# Dashboard
python -m zebrav1.dashboard  # http://0.0.0.0:8000

# All flags
python -m zebrav1.gym_env.demo --brain --render --monitor --record --sound --multi-agent --spiking --autosave --steps 1000
```
