#!/bin/bash

# Setup script for RL Cluster Benchmark

set -e

echo "=========================================="
echo "RL Cluster Benchmark - Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Check if Python version is compatible
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 12 ]; then
    echo ""
    echo "⚠️  WARNING: Python $python_version detected"
    echo "Procgen currently supports Python 3.8-3.11"
    echo ""
    echo "Options:"
    echo "  1. Use Python 3.11 (recommended)"
    echo "  2. Try installing procgen from source (experimental)"
    echo ""
    read -p "Continue with source installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Please install Python 3.11 and run this script again:"
        echo "  sudo apt install python3.11 python3.11-venv  # Ubuntu/Debian"
        echo "  brew install python@3.11  # macOS"
        echo ""
        echo "Then create the environment with Python 3.11:"
        echo "  python3.11 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
    USE_SOURCE_INSTALL=1
else
    USE_SOURCE_INSTALL=0
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies..."

if [ "$USE_SOURCE_INSTALL" -eq 1 ]; then
    echo "Installing dependencies except procgen..."
    pip install torch>=2.0.0
    pip install gymnasium>=0.29.0
    pip install numpy>=1.24.0
    pip install tensorboard>=2.13.0
    pip install pyyaml>=6.0
    pip install tqdm>=4.65.0
    pip install matplotlib>=3.7.0
    pip install psutil>=5.9.0
    
    echo ""
    echo "Attempting to install procgen from source..."
    pip install cmake
    
    # Try to install procgen from GitHub
    if pip install git+https://github.com/openai/procgen.git 2>/dev/null; then
        echo "✓ Successfully installed procgen from source"
    else
        echo ""
        echo "⚠️  Failed to install procgen from source"
        echo ""
        echo "Alternative options:"
        echo "  1. Use Docker with Python 3.11 (recommended for production)"
        echo "  2. Install Python 3.11 system-wide"
        echo "  3. Use conda to manage Python versions"
        echo ""
        echo "For now, you can still use the framework without procgen"
        echo "by using other gymnasium environments."
    fi
else
    pip install -r requirements.txt
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run a quick test, run:"
echo "  python quick_test.py"
echo ""
echo "To start training, run:"
echo "  python train_ppo_sequential.py"
echo ""
echo "Or use the Makefile:"
echo "  make test         # Run tests"
echo "  make train        # Start training"
echo "  make tensorboard  # View training progress"
echo ""
echo "=========================================="

