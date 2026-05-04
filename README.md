# Bark Detector

Real-time dog bark detection running on a Raspberry Pi Zero 2W.  Audio is
captured through a Behringer UM2 USB interface connected to a Pronomic DM-58-B
XLR microphone, classified by the YAMNet neural network (TFLite), and logged as
time-series data to InfluxDB Cloud for Grafana dashboards.

```
Pronomic DM-58-B (XLR)
    → Behringer UM2 (USB audio interface)
        → Raspberry Pi Zero 2W  ←  this project
            → InfluxDB Cloud (EU Frankfurt)
                → Grafana dashboard
```

---

## Features

- **~1.5–2 s first-detection latency** — VAD buffer fill + 0.96 s context window + YAMNet inference
- **RMS Voice Activity Detection** — skips inference during silence, saving ~80 % of CPU
- **YAMNet TFLite** — pre-trained on 521 AudioSet classes, inference takes 80–120 ms on Pi Zero 2W (not the total detection delay — see Latency Budget)
- **Debounce** — configurable cooldown prevents duplicate events during a single bark episode
- **InfluxDB Cloud integration** — structured time-series data with spectral band features
- **Local CSV fallback** — every event is written locally, even when the network is down
- **WAV snippet ring buffer** — saves the last N bark audio clips for model fine-tuning
- **Real-time config updates** — change thresholds via Upstash Redis without restarting
- **systemd service** — auto-starts on boot, restarts on crash

---

## Project Structure

```
bark_detector/
├── bark_detector.py        Main event loop and orchestration
├── audio_pipeline.py       PyAudio stream, VAD, ring buffer
├── classifier.py           YAMNet TFLite wrapper
├── influx_logger.py        InfluxDB Cloud writer + CSV fallback
├── redis_config.py         Real-time config polling via Upstash Redis
├── install.sh              One-shot setup script for the Pi
├── requirements.txt        Python dependencies
├── bark-detector.service   systemd unit file
├── config/
│   └── settings.json       All tuneable parameters and credentials
├── models/
│   ├── yamnet.tflite        YAMNet TFLite model (downloaded by install.sh)
│   └── yamnet_classes.csv   AudioSet class label map
├── logs/
│   └── bark_events.csv      Local CSV backup of all bark events
├── snippets/               WAV ring buffer of recent bark audio
└── docs/
    └── implementation_plan.md  Full implementation plan
```

---

## Quick Start

### 1. Flash the Pi

Use **Raspberry Pi OS Lite (64-bit)**.  In Raspberry Pi Imager's advanced
settings, enable SSH, set your WiFi credentials, and set the hostname to
`barkpi`.

### 2. SSH into the Pi

```bash
ssh pi@barkpi.local
```

### 3. Clone and run the setup script

```bash
git clone <your-repo-url> ~/bark_detector
cd ~/bark_detector
bash install.sh
```

The script installs system packages, creates a virtualenv at `~/bark_env`,
downloads the YAMNet model, and writes a template `config/settings.json`.

### 4. Verify the audio interface

```bash
arecord -l
# Expected: "card 1: UM2 [Behringer UM2], device 0: USB Audio [USB Audio]"
```

If the UM2 shows up on a different card number, update `audio.device_index`
in `config/settings.json`.

### 5. Enter your InfluxDB credentials

```bash
nano config/settings.json
```

Fill in `influxdb.token`, `influxdb.org`, and optionally `influxdb.url` if
you chose a different region.

### 6. Run the detector

```bash
source ~/bark_env/bin/activate
python bark_detector.py
```

Press `Ctrl+C` to stop.

### 7. Install as a systemd service (autostart on boot)

```bash
sudo cp bark-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bark-detector
sudo systemctl status bark-detector
```

---

## Configuration Reference

All settings live in `config/settings.json`.

| Section | Key | Default | Description |
|---|---|---|---|
| `audio` | `device_index` | `1` | ALSA card number for the Behringer UM2 |
| `audio` | `sample_rate` | `16000` | Sample rate in Hz (YAMNet requires 16 kHz) |
| `audio` | `chunk_ms` | `200` | Duration of each read chunk in milliseconds |
| `vad` | `rms_threshold` | `0.015` | Minimum RMS energy to trigger VAD |
| `vad` | `min_chunks_above` | `2` | Consecutive loud chunks before VAD fires |
| `classifier` | `confidence_threshold` | `0.82` | YAMNet score required to log a bark event |
| `classifier` | `debounce_seconds` | `1.5` | Cooldown between consecutive bark events |
| `influxdb` | `token` | *(required)* | InfluxDB API token |
| `influxdb` | `org` | *(required)* | InfluxDB organisation name |
| `logging` | `max_snippets` | `500` | Maximum WAV files kept in the snippets folder |

### Tuning Sensitivity

| Symptom | Cause | Fix |
|---|---|---|
| Too many false positives | Threshold too low | Raise `confidence_threshold` to `0.88` |
| Missing real barks | Threshold too high | Lower `confidence_threshold` to `0.75` |
| High CPU load | VAD not filtering | Raise `rms_threshold` to `0.025` |
| Crackling audio | Buffer overflow | Raise `chunk_ms` to `400` |
| Rapid duplicate events | Debounce too short | Raise `debounce_seconds` to `3.0` |

---

## InfluxDB Schema

**Measurement:** `bark_event`

| Type | Name | Description |
|---|---|---|
| Tag | `dog_id` | Identifier for the dog source, e.g. `neighbor_unknown` |
| Tag | `model_version` | Classifier version, e.g. `yamnet_v1` |
| Tag | `location` | Physical location tag, e.g. `garden` |
| Field | `confidence` | Max YAMNet bark-class score [0, 1] |
| Field | `rms_db` | RMS energy in decibels |
| Field | `band_0` … `band_4` | Spectral energy bands (low → high frequency) |

### Example Flux Query (barks per hour)

```flux
from(bucket: "barks")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "bark_event")
  |> filter(fn: (r) => r._field == "confidence")
  |> aggregateWindow(every: 1h, fn: count)
```

---

## Real-Time Config via Upstash Redis (Optional)

Set the following keys in your Upstash Redis instance to update thresholds
without restarting the service.  Changes take effect within 60 seconds.

| Redis Key | Type | Example |
|---|---|---|
| `bark:confidence_threshold` | float | `0.85` |
| `bark:rms_threshold` | float | `0.020` |
| `bark:debounce_seconds` | float | `2.0` |
| `bark:dog_id` | string | `rex_neighbor` |

Leave `redis.url` as the placeholder value in `settings.json` to disable this
feature entirely.

---

## Latency Budget

This table shows the time from when a bark starts until it is first detected.

| Step | Time | Notes |
|---|---|---|
| VAD fill (2 loud chunks required) | ~400 ms | `min_chunks_above: 2` × 200 ms chunks |
| Context window fill (0.96 s buffer) | ~960 ms | YAMNet requires exactly 15 360 samples |
| YAMNet TFLite inference | ~100 ms | Per-iteration cost on Pi Zero 2W |
| InfluxDB write + CSV | ~50 ms | Only on confirmed bark events |
| **First detection after bark starts** | **~1.5–2 s** | Dominated by buffer fill, not inference |

The 80–120 ms inference figure describes how long the neural network takes to
process one window — it is not the end-to-end detection delay.  Subsequent barks
within an ongoing episode are detected faster because the ring buffer is already
populated; only the VAD hysteresis (400 ms) and inference (~100 ms) apply.

---

## Monitoring

```bash
# Follow live logs via journalctl
journalctl -u bark-detector -f

# View the local CSV backup
cat logs/bark_events.csv

# Count total barks logged
wc -l logs/bark_events.csv

# List recent audio snippets
ls -lth snippets/ | head -20
```

---

## Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi Zero 2W | Edge inference host |
| Behringer UM2 | USB audio interface (phantom power for XLR mic) |
| Pronomic DM-58-B | Dynamic XLR microphone (outdoor-facing) |

---

## License

MIT
