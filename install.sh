#!/usr/bin/env bash
# install.sh — One-shot setup script for the Bark Detector on Raspberry Pi Zero 2W
#
# Run this script once after cloning the repository onto the Pi:
#   bash install.sh
#
# What this script does:
#   1. Installs required system packages (PortAudio, ALSA, wget, bzip2).
#   2. Installs Miniconda at ~/miniconda3 if it is not already present.
#   3. Creates or updates the Conda environment from environment.yml.
#   4. Downloads the YAMNet TFLite model and class labels into models/.
#   5. Creates the required directory structure (logs, snippets, config).
#   6. Writes a template config/settings.json if one does not already exist.
#   7. Prints next-step instructions.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh

set -euo pipefail  # exit on error, undefined variable, or pipe failure

# ---------------------------------------------------------------------------
# Colour helpers for readable output
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # no colour

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
info "Updating package lists and installing system dependencies..."
sudo apt update -q
sudo apt install -y \
    git \
    portaudio19-dev \
    python3-dev \
    libasound2-dev \
    alsa-utils \
    wget \
    bzip2

# ---------------------------------------------------------------------------
# 2. Miniconda installation
# ---------------------------------------------------------------------------
CONDA_DIR="$HOME/miniconda3"
CONDA_BIN="$CONDA_DIR/bin/conda"
MINICONDA_INSTALLER="/tmp/miniconda.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"

if [ -x "$CONDA_BIN" ]; then
    warning "Miniconda already exists at $CONDA_DIR — skipping installer download."
else
    info "Downloading Miniconda installer..."
    wget -q --show-progress "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"

    info "Installing Miniconda to $CONDA_DIR..."
    bash "$MINICONDA_INSTALLER" -b -p "$CONDA_DIR"
fi

# Load Conda into this shell
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 3. Conda environment
# ---------------------------------------------------------------------------
ENV_NAME="bark-detector"
ENV_FILE="environment.yml"

if [ ! -f "$ENV_FILE" ]; then
    error "Missing $ENV_FILE in $(pwd)"
    exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    info "Updating existing Conda environment: $ENV_NAME"
    "$CONDA_BIN" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    info "Creating Conda environment: $ENV_NAME"
    "$CONDA_BIN" env create -f "$ENV_FILE"
fi

info "Conda environment ready."

# ---------------------------------------------------------------------------
# 4. YAMNet model download
# ---------------------------------------------------------------------------
MODELS_DIR="$(pwd)/models"
mkdir -p "$MODELS_DIR"

TFLITE_MODEL="$MODELS_DIR/yamnet.tflite"
CLASSES_CSV="$MODELS_DIR/yamnet_classes.csv"

if [ -f "$TFLITE_MODEL" ]; then
    warning "YAMNet model already exists — skipping download."
else
    info "Downloading YAMNet TFLite model (~3 MB)..."
    wget -q --show-progress \
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/lite-model_yamnet_tflite_1.tflite" \
        -O "$TFLITE_MODEL"
    info "YAMNet model downloaded."
fi

if [ -f "$CLASSES_CSV" ]; then
    warning "YAMNet class labels already exist — skipping download."
else
    info "Downloading YAMNet class label CSV..."
    wget -q --show-progress \
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv" \
        -O "$CLASSES_CSV"
    info "Class labels downloaded."
fi

# ---------------------------------------------------------------------------
# 5. Directory structure
# ---------------------------------------------------------------------------
info "Creating project directories..."
mkdir -p logs snippets config

# ---------------------------------------------------------------------------
# 6. Template settings.json (only if missing)
# ---------------------------------------------------------------------------
CONFIG_FILE="config/settings.json"

if [ -f "$CONFIG_FILE" ]; then
    warning "config/settings.json already exists — skipping template creation."
else
    info "Creating template config/settings.json..."
    cat > "$CONFIG_FILE" << 'EOF'
{
  "audio": {
    "device_index": 1,
    "sample_rate": 16000,
    "channels": 1,
    "chunk_ms": 200,
    "format": "int16"
  },
  "vad": {
    "rms_threshold": 0.015,
    "min_chunks_above": 2
  },
  "classifier": {
    "model_path": "models/yamnet.tflite",
    "bark_class_index": 74,
    "confidence_threshold": 0.82,
    "debounce_seconds": 1.5
  },
  "influxdb": {
    "url": "https://eu-central-1-1.aws.cloud2.influxdata.com",
    "token": "YOUR_TOKEN_HERE",
    "org": "YOUR_ORG_HERE",
    "bucket": "barks"
  },
  "redis": {
    "url": "rediss://eu1-xxx.upstash.io:6380",
    "password": "YOUR_REDIS_PASSWORD_HERE"
  },
  "logging": {
    "snippet_duration_s": 2.0,
    "max_snippets": 500,
    "local_csv_backup": true
  }
}
EOF
    info "Template config written. Edit config/settings.json to add your InfluxDB credentials."
fi

# ---------------------------------------------------------------------------
# 7. Summary and next steps
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify your Behringer UM2 is detected:"
echo "       arecord -l"
echo ""
echo "  2. Edit config/settings.json and fill in your InfluxDB token + org."
echo ""
echo "  3. Test a 5-second recording:"
echo "       arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/test.wav"
echo "       aplay /tmp/test.wav"
echo "     If the device warns that it is really recording at 44100 Hz, retry with:"
echo "       arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/test.wav"
echo ""
echo "  4. Run the detector:"
echo "       source ~/miniconda3/etc/profile.d/conda.sh"
echo "       conda activate bark-detector"
echo "       python bark_detector.py"
echo ""
echo "  5. (Optional) Install as a systemd service:"
echo "       sudo cp bark-detector.service /etc/systemd/system/"
echo "       sudo systemctl enable --now bark-detector"
echo ""
