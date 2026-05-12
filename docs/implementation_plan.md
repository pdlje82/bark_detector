# Bark Detector — Full Implementation Plan

## Hardware Setup (Reference)
```
Pronomic DM-58-B (XLR) → Behringer UM2 → USB-B/microUSB → Pi Zero 2W
                                                              └── WiFi → InfluxDB Cloud
```

---

## Phase 0 — Prepare the Pi (one-time, manual ~15 min)

### 0.1 Flash Raspberry Pi OS
1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Choose OS: **Raspberry Pi OS Lite (64-bit)** — no desktop needed
3. Before flashing: click the gear icon → Advanced Settings:
   - Hostname: `barkpi`
   - Enable SSH ✅
   - Enter WiFi SSID + password
   - Username: `pi` / set a password
4. Flash to microSD, insert into Pi, power on

### 0.2 Test SSH Connection
```bash
# From your laptop (after ~60s boot time):
ssh pi@barkpi.local

# If unreachable:
ssh pi@<IP-address>  # look up IP in your router
```

### 0.3 Update the System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-pip python3-venv portaudio19-dev \
                    python3-dev libasound2-dev alsa-utils
```

---

## Phase 1 — Verify Audio Hardware

```bash
# Is the UM2 recognized?
arecord -l
# Expected output: "card 1: UM2 [Behringer UM2], device 0: USB Audio [USB Audio]"

# Quick test: record 5 seconds and play back
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -d 5 test.wav
aplay test.wav

# Check levels (bark/clap into the mic, watch for peaks)
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -vv /dev/null
```

If the raw `hw:1,0` device warns that the requested `16000 Hz` rate was mapped
to a native hardware rate such as `44100 Hz`, switch the verification command
to `plughw:1,0` instead.  Some USB microphones only expose fixed sample rates,
and `plughw` lets ALSA insert the required software conversion:

```bash
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 5 test.wav
```

### ALSA Configuration (set UM2 as default device)
```bash
sudo nano /etc/asound.conf
```
```
pcm.!default {
    type hw
    card 1
}
ctl.!default {
    type hw
    card 1
}
```

---

## Phase 2 — Set Up Python Environment

```bash
cd ~
python3 -m venv bark_env
source bark_env/bin/activate

pip install --upgrade pip
pip install pyaudio numpy scipy
pip install tflite-runtime          # YAMNet inference
pip install influxdb-client         # InfluxDB Cloud
pip install redis                   # Upstash Redis (real-time config)
pip install requests                # Telegram notifications (optional)
```

### Download YAMNet TFLite Model
```bash
mkdir -p ~/bark_detector/models
cd ~/bark_detector/models

# YAMNet from TensorFlow Hub (TFLite version)
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/lite-model_yamnet_tflite_1.tflite \
     -O yamnet.tflite

# Download class labels
wget https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv \
     -O yamnet_classes.csv
```

---

## Phase 3 — Project Structure

```bash
mkdir -p ~/bark_detector/{models,logs,snippets,config}
cd ~/bark_detector
```

```
bark_detector/
├── models/
│   ├── yamnet.tflite
│   └── yamnet_classes.csv
├── config/
│   └── settings.json          ← local configuration
├── logs/                      ← local CSV backups
├── snippets/                  ← WAV ring buffer for training
├── bark_detector.py           ← main program
├── audio_pipeline.py          ← audio capture + VAD + features
├── classifier.py              ← YAMNet TFLite wrapper
├── influx_logger.py           ← InfluxDB Cloud integration
├── redis_config.py            ← real-time config via Upstash
└── install.sh                 ← setup script
```

---

## Phase 4 — Core Code

See the individual source files in the project root for the full implementation.

### 4.1 `config/settings.json`
Central configuration file. All thresholds, credentials, and tuning parameters live here.
Edit this file to adjust sensitivity, swap credentials, or change the audio device index.

### 4.2 `audio_pipeline.py`
Handles PyAudio stream setup, chunk reading, RMS-based Voice Activity Detection (VAD),
and assembly of 0.96-second context windows for YAMNet.

### 4.3 `classifier.py`
Thin TFLite wrapper around the YAMNet model. Loads the interpreter once, then runs
`predict()` on each context window. Returns per-class confidence scores.

### 4.4 `influx_logger.py`
Writes bark events (confidence, RMS, spectral bands, timestamps) to InfluxDB Cloud
using the synchronous write API. Falls back to a local CSV if the network is unavailable.

### 4.5 `bark_detector.py`
Main event loop: read chunk → VAD → debounce → classify → log. Handles graceful
shutdown via SIGTERM (systemd) and KeyboardInterrupt.

---

## Phase 5 — Set Up systemd Service (Autostart)

```bash
sudo nano /etc/systemd/system/bark-detector.service
```

```ini
[Unit]
Description=Bark Detector
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/bark_detector
Environment=PATH=/home/pi/bark_env/bin:/usr/bin:/bin
ExecStart=/home/pi/bark_env/bin/python bark_detector.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable bark-detector
sudo systemctl start bark-detector

# Check status
sudo systemctl status bark-detector

# Follow live logs
journalctl -u bark-detector -f
```

---

## Phase 6 — Set Up InfluxDB Cloud

### 6.1 Create Account
1. Go to [cloud2.influxdata.com](https://cloud2.influxdata.com) → Sign Up
2. Region: **EU Central (Frankfurt)**
3. Plan: **Free**

### 6.2 Create Token + Bucket
```
Data → Buckets → Create Bucket → Name: "barks", Retention: 30d
Data → API Tokens → Generate → All Access Token → copy
```

### 6.3 Enter Token in settings.json
```bash
nano ~/bark_detector/config/settings.json
# Fill in influxdb.token and influxdb.org
```

### 6.4 Test the Connection
```bash
cd ~/bark_detector && source ~/bark_env/bin/activate
python -c "
from influxdb_client import InfluxDBClient
import json
cfg = json.load(open('config/settings.json'))['influxdb']
client = InfluxDBClient(url=cfg['url'], token=cfg['token'], org=cfg['org'])
print('Connected:', client.ping())
"
```

---

## Phase 7 — Grafana Dashboard (optional but recommended)

1. Go to [grafana.com](https://grafana.com) → Free Account
2. Connections → Add InfluxDB → Flux Query Language
3. Enter InfluxDB URL + Token
4. Create a dashboard with:

```flux
// Barks per hour
from(bucket: "barks")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "bark_event")
  |> filter(fn: (r) => r._field == "confidence")
  |> aggregateWindow(every: 1h, fn: count)
```

---

## Latency Analysis

```
Read audio chunk:          200ms  (chunk_ms setting)
VAD check:                  <1ms  (RMS calculation)
Build context audio:        <1ms  (np.concatenate)
YAMNet inference:         ~100ms  (Pi Zero 2W, TFLite)
Decision + logging:        ~50ms  (InfluxDB write)
─────────────────────────────────
TOTAL:                    ~351ms  ✅ well under 1s
```

**Tuning options if needed:**
- `chunk_ms: 100` → halves the chunk wait time
- Raise `rms_threshold` → fewer inferences
- Write InfluxDB asynchronously → non-blocking logging

---

## Deployment with Claude Code

```bash
# 1. SSH from your laptop
ssh pi@barkpi.local

# 2. Start Claude Code directly on the Pi
claude

# 3. Claude Code takes over from here:
# - write files
# - install dependencies
# - test audio
# - debug errors
# - set up the service
```

---

## Troubleshooting Cheatsheet

| Problem | Diagnosis | Fix |
|---|---|---|
| UM2 not recognized | `arecord -l` | Replug USB, check `lsusb` |
| Too many false positives | Confidence threshold too low | Raise `confidence_threshold` to 0.88 |
| Too many missed detections | Confidence threshold too high | Lower threshold to 0.75 |
| InfluxDB timeout | Network issue | Check CSV backup, validate token |
| High CPU load | VAD not filtering | Raise `rms_threshold` |
| Crackling audio | Buffer overflow | Increase `chunk_size` |

---

*Target: Pi Zero 2W + Behringer UM2 + Pronomic DM-58-B + InfluxDB Cloud*
*Target latency: < 400ms | Architecture: VAD → YAMNet TFLite → InfluxDB*
