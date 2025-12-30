#!/bin/bash

# Setup script for RL Cluster Benchmark with Python 3.11
# Use this if you have Python 3.12+ and need procgen support

set -e

echo "=========================================="
echo "RL Cluster Benchmark - Python 3.11 Setup"
echo "=========================================="
echo ""

# Check if python3.11 is available
if command -v python3.11 &> /dev/null; then
    echo "✓ Found Python 3.11"
    PYTHON_CMD=python3.11
elif command -v python3.10 &> /dev/null; then
    echo "✓ Found Python 3.10 (also compatible)"
    PYTHON_CMD=python3.10
elif command -v python3.9 &> /dev/null; then
    echo "✓ Found Python 3.9 (also compatible)"
    PYTHON_CMD=python3.9
else
    echo "✗ Python 3.9-3.11 not found"
    echo ""
    echo "Please install Python 3.11:"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt update"
    echo "  sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo ""
    echo "Fedora:"
    echo "  sudo dnf install python3.11 python3.11-devel"
    echo ""
    echo "macOS (Homebrew):"
    echo "  brew install python@3.11"
    echo ""
    echo "Or use pyenv:"
    echo "  pyenv install 3.11.9"
    echo "  pyenv local 3.11.9"
    exit 1
fi

python_version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Using Python $python_version"
echo ""

# Remove old venv if it exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create virtual environment with Python 3.11
echo "Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete with Python $python_version!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python quick_test.py"
echo ""
echo "To start training:"
echo "  python train_ppo_sequential.py"
echo ""
echo "=========================================="

