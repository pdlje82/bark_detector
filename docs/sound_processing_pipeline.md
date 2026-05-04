# Sound Processing Pipeline — From Bark to Detection

This document traces exactly what happens to a sound between the moment a dog
barks outside and the moment a bark event is written to InfluxDB.  Each stage
is explained in terms of what the signal looks like, why the step exists, and
how long it takes.

---

## Overview

```
Dog barks
    │
    ▼
[1] Microphone (Pronomic DM-58-B)
    │  converts air pressure waves → electrical voltage
    ▼
[2] Behringer UM2 — Preamp + ADC
    │  amplifies signal, samples at 16 000 Hz, sends USB audio frames to Pi
    ▼
[3] PyAudio / PortAudio — chunk read (200 ms)
    │  accumulates 3 200 samples, hands buffer to Python
    ▼
[4] Normalisation
    │  int16 → float32, scaled to [-1.0, 1.0]
    ▼
[5] Ring Buffer
    │  stores the last 1 second of chunks (5 chunks × 3 200 samples)
    ▼
[6] Voice Activity Detection (VAD)
    │  RMS energy check — silent chunks are discarded here
    ▼
[7] Context Window Assembly
    │  concatenate ring buffer → 15 360-sample (0.96 s) window
    ▼
[8] YAMNet TFLite Inference
    │  waveform → mel spectrogram → MobileNet → 521 class scores
    ▼
[9] Threshold Decision
    │  max bark-class score ≥ confidence_threshold?
    ▼
[10] Event Logging
       InfluxDB Cloud write + local CSV backup + WAV snippet save
```

---

## Stage 1 — Microphone: Pressure Waves → Voltage

The Pronomic DM-58-B is a **dynamic XLR microphone**.  A thin diaphragm
vibrates in response to air pressure changes caused by the bark.  A coil
attached to the diaphragm moves inside a magnetic field, generating a small
alternating voltage proportional to the sound pressure level.

- Signal level at this point: millivolts, unbalanced
- Frequency range of a dog bark: roughly 300 Hz – 4 000 Hz (fundamental +
  harmonics); YAMNet uses up to 8 000 Hz (Nyquist limit of 16 kHz sampling)

---

## Stage 2 — Behringer UM2: Preamp + Analogue-to-Digital Conversion

The UM2 performs two jobs:

**Preamp** — Boosts the microphone's millivolt signal to line level (roughly
1 V peak-to-peak) using a low-noise amplifier.  The gain knob on the UM2
controls this stage; setting it too high introduces clipping, too low causes
the signal to drown in quantisation noise.

**ADC (Analogue-to-Digital Converter)** — Samples the amplified voltage
16 000 times per second and encodes each sample as a signed 16-bit integer
(range −32 768 to +32 767).  This produces a stream of USB audio frames that
the Pi reads via PortAudio.

- Sample rate: **16 000 Hz** (required by YAMNet; also efficient on Pi Zero 2W)
- Bit depth: **16-bit PCM**
- Channels: **1 (mono)**

---

## Stage 3 — PyAudio Chunk Read (~200 ms)

`AudioPipeline.read_chunk()` calls `stream.read(3200)`, which **blocks** until
PortAudio has accumulated exactly 3 200 samples (= 200 ms at 16 kHz).

This blocking read is the heartbeat of the entire pipeline.  Nothing else
happens until the chunk is ready.  The chunk size is a trade-off:

| Smaller chunks (e.g. 100 ms) | Larger chunks (e.g. 400 ms) |
|---|---|
| Lower latency to VAD | Higher latency to VAD |
| More CPU overhead (more loop iterations) | Less CPU overhead |
| More susceptible to USB jitter | More tolerant of USB jitter |

200 ms is a reasonable default for the Pi Zero 2W.

---

## Stage 4 — Normalisation: int16 → float32

```python
audio_float32 = audio_int16.astype(np.float32) / 32768.0
```

The raw int16 samples are converted to float32 and divided by 32 768 (the
maximum positive value of a signed 16-bit integer), mapping the range to
approximately [−1.0, 1.0].

YAMNet expects normalised floating-point waveform input.  Keeping the values
in this range also makes the RMS threshold in Stage 6 independent of the
specific ADC bit depth.

---

## Stage 5 — Ring Buffer

The normalised chunk is appended to a `collections.deque` with a fixed maximum
length of 5 chunks (= 1 second of audio).  When the deque is full, appending a
new chunk automatically discards the oldest one.

```
Time →

Chunk 0  [████████████████████]  oldest, will be discarded next
Chunk 1  [████████████████████]
Chunk 2  [████████████████████]
Chunk 3  [████████████████████]
Chunk 4  [████████████████████]  newest (just read)

└── concatenated = 16 000 samples = 1.0 second of audio
```

The ring buffer decouples the chunk-read rate from the classifier's input
requirement.  YAMNet needs 0.96 s; the buffer always has slightly more,
so assembly is just an array slice.

---

## Stage 6 — Voice Activity Detection (VAD)

```python
rms = sqrt( mean( audio² ) )
```

**Root Mean Square (RMS)** is the standard measure of signal power.  For a
pure sine wave it equals the amplitude divided by √2; for a bark, it is
higher and more variable than for background silence.

The VAD uses a **hysteresis counter** to avoid reacting to single transient
spikes (a distant car door, a camera click):

```
Chunk loud (RMS > threshold)?  →  counter += 1
Chunk quiet?                   →  counter -= 1  (min 0)
counter ≥ min_chunks_above?    →  activity declared, proceed to inference
```

With `min_chunks_above: 1` (current setting), one loud 100 ms chunk is enough
to trigger inference.  This minimises latency but removes the transient-spike
filter — a single loud non-bark sound (door slam, dropped object) can now reach
the classifier.  The raised `confidence_threshold: 0.85` compensates for this.

With `min_chunks_above: 2` and `chunk_ms: 200` (original defaults), two
consecutive loud chunks (400 ms total) were required, filtering most transients
at the cost of higher latency.

**Effect on latency:** With current settings the VAD adds a minimum of 100 ms
from the start of the bark before inference can begin.

---

## Stage 7 — Context Window Assembly

YAMNet requires **exactly 0.96 seconds = 15 360 samples** at 16 kHz.  The
ring buffer holds 16 000 samples (1.0 s), so the context window is simply the
last 15 360 samples of the concatenated buffer:

```python
buffered = np.concatenate(list(ring_buffer))  # 16 000 samples
context  = buffered[-15360:]                  # last 0.96 s
```

If the buffer is not yet full (first few seconds after startup), the beginning
is zero-padded.  Zero-padding introduces mild classification noise at startup
but does not affect steady-state operation.

### The 0.96 s window does not add latency — but it does affect confidence

A common misconception is that the 0.96-second window forces the detector to
wait 960 ms before it can classify.  **This is not the case.**  The ring buffer
is filled continuously while the detector runs, so by the time a bark happens
the buffer already contains ~1 second of pre-bark audio.  The window is
assembled instantly from that historical data — it costs zero extra latency.

What the window *does* affect is **how much of the input actually contains bark
audio** on the first detection attempt:

```
First detection (100 ms after bark starts, with current settings):

|←──────── 860 ms of pre-bark silence/ambient noise ────────→|← 100 ms bark →|
 ────────────────────────────────────────────────────────────────────────────
                           0.96 s YAMNet input
```

Only ~10 % of the window is bark at this point.  YAMNet was trained on
AudioSet clips where the target sound typically fills the majority of the clip,
so a bark buried in 860 ms of silence may produce a **lower confidence score**
than the same bark heard on the second or third chunk, when it occupies a
larger fraction of the window:

```
Second detection attempt (200 ms after bark starts):

|←── 760 ms ambient ──→|←──────── 200 ms of bark ────────→|
 ─────────────────────────────────────────────────────────
                    0.96 s YAMNet input  (bark = ~21 %)

Third detection attempt (300 ms after bark starts):

|←── 660 ms ambient ──→|←──────── 300 ms of bark ────────→|
 ─────────────────────────────────────────────────────────
                    0.96 s YAMNet input  (bark = ~31 %)
```

**Practical consequence:** For short or quiet barks, the very first inference
(at 100 ms) may score below `confidence_threshold` and be missed, with the
actual detection firing on the 2nd or 3rd chunk (200–300 ms after the bark
starts).  For loud or sustained barks — the typical case with a dog barking
repeatedly — the first inference is usually sufficient.

**The only structural fix** would be to replace YAMNet with a model designed
for shorter input windows (e.g. 100–200 ms), which would require retraining or
finding an alternative pre-trained model.  For the current use case this
trade-off is acceptable.

---

## Stage 8 — YAMNet TFLite Inference (~80–120 ms)

YAMNet's internal processing chain:

```
15 360-sample waveform (0.96 s @ 16 kHz)
    │
    ▼  Short-Time Fourier Transform (STFT)
    │  window: 25 ms, hop: 10 ms → 96 time frames
    │
    ▼  Mel filterbank (64 mel bins, 125 Hz – 7 500 Hz)
    │  maps linear frequency bins to perceptual mel scale
    │
    ▼  Log compression
    │  log(mel_spectrogram + small_epsilon)
    │  → input shape: [1, 96, 64] = [batch, time_frames, mel_bins]
    │
    ▼  MobileNet v1 (depthwise-separable convolutions)
    │  lightweight CNN designed for mobile/edge inference
    │
    ▼  521-element softmax output vector
       one probability score per AudioSet class
```

The TFLite runtime runs this entire chain in **one synchronous call**
(`interpreter.invoke()`).  On the Pi Zero 2W's ARM Cortex-A53 cores, this
takes 80–120 ms.

The relevant output indices for dog sounds:

| Index | Label     |
|-------|-----------|
| 74    | Dog bark  |
| 75    | Bow-wow   |
| 76    | Growling  |
| 503   | Animal    |

---

## Stage 9 — Threshold Decision

```python
bark_confidence = max(scores[74], scores[75], scores[76], scores[503])

if bark_confidence >= confidence_threshold:   # default: 0.82
    on_bark(...)
```

The maximum score across all bark-related classes is compared to the configured
threshold.  Using the maximum rather than, say, `scores[74]` alone makes
detection more robust: a growl or repeated bow-wow that YAMNet labels
differently from "Dog bark" can still trigger an event.

**Debounce:** After a positive detection, the threshold check is suppressed for
`debounce_seconds` (default 1.5 s).  This means a 10-second barking episode
produces roughly one event every 1.5 s rather than one per 200 ms chunk, which
keeps InfluxDB write volume manageable and prevents log flooding.

---

## Stage 10 — Event Logging

Three things happen in parallel when a bark is confirmed:

1. **InfluxDB Cloud write** — a `bark_event` point with confidence, RMS dB,
   5 spectral energy bands, dog_id tag, and UTC timestamp.

2. **Local CSV append** — the same values are written to `logs/bark_events.csv`
   unconditionally, serving as a fallback if InfluxDB is unreachable.

3. **WAV snippet save** — the 0.96-second context window is written to
   `snippets/bark_YYYYMMDD_HHMMSS_ffffff.wav` for later manual review and
   model fine-tuning.  A ring buffer of 500 files is maintained; the oldest
   file is deleted when the limit is exceeded.

---

## End-to-End Timing

```
t = 0 ms      Dog starts barking
t = 0–200 ms  Chunk 1 captured (200 ms) — VAD counter: 1 of 2
t = 200–400 ms Chunk 2 captured (200 ms) — VAD counter: 2 of 2 ✓ VAD fires
t = 400 ms    Context window assembled (15 360 samples from ring buffer)
t = 400–500 ms YAMNet inference (~100 ms)
t = 500–550 ms InfluxDB write (~50 ms)
──────────────────────────────────────────────────────────────────────
t ≈ 550 ms    First detection (optimistic — assumes ring buffer already full)

If the ring buffer is not yet full (startup):
    Buffer fill time adds up to 960 ms → first detection at ~1.5 s

Worst case (cold start + slow network write):
    ~1.5–2 s from bark start to InfluxDB data point
```

**Note:** The 0.96-second context window can overlap with the VAD fill period
— the ring buffer is being populated continuously, so by the time the VAD
counter reaches 2, the buffer already contains recent audio.  The limiting
factor is the VAD hysteresis (400 ms), not the buffer fill, in steady state.

---

## Tuning Reference

| Goal | Parameter | Direction |
|---|---|---|
| Faster first detection | `min_chunks_above` | Lower (minimum: 1) |
| Fewer false positives | `confidence_threshold` | Raise (e.g. 0.88) |
| Less CPU usage | `rms_threshold` | Raise (filter more silence) |
| Fewer duplicate events | `debounce_seconds` | Raise (e.g. 3.0) |
| Shorter audio window | `chunk_ms` | Lower (increases CPU overhead) |
