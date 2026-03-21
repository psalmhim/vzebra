#!/bin/bash
# Migration script: setup vzebra on a new GPU machine
# Usage: bash migrate.sh

set -e

echo "================================================"
echo "vzebra GPU Migration Setup"
echo "================================================"

# Check GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')" 2>/dev/null || echo "PyTorch not installed yet"

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Verify
echo ""
echo "Verification:"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  MPS: Available (Apple Silicon)')
else:
    print(f'  Device: CPU only')
"

# Check if weights exist
echo ""
if [ -f "zebrav1/weights/classifier.pt" ]; then
    echo "Weights found — ready to run"
else
    echo "No weights found — training needed"
    echo ""
    echo "Run training pipeline:"
    echo "  python -m zebrav1.tests.step8_genomic_pretraining"
    echo "  python -m zebrav1.tests.step10_hebbian_finetuning"
    echo "  python -m zebrav1.tests.step11_object_classification"
    echo "  python -m zebrav1.tests.step11b_gameplay_classifier"
    echo "  python -m zebrav1.tests.step26_wfb_pe_training"
    echo "  python -m zebrav1.tests.step43_online_rl"
fi

echo ""
echo "================================================"
echo "Ready! Commands:"
echo "  source .venv/bin/activate"
echo "  python -m zebrav1.dashboard          # Web dashboard"
echo "  python -m zebrav1.tests.step43_online_rl  # RL training"
echo "================================================"
