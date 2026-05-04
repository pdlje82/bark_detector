#!/usr/bin/env python3
"""
bark_detector.py — Main Entry Point for the Bark Detector

System overview:
    Pronomic DM-58-B (XLR microphone)
        → Behringer UM2 (USB audio interface)
            → Raspberry Pi Zero 2W (this program)
                → InfluxDB Cloud (time-series storage)
                    → Grafana (dashboard)

Processing pipeline per iteration (target latency budget):
    1. Read audio chunk from UM2          ~200 ms  (chunk_ms setting)
    2. RMS Voice Activity Detection         < 1 ms  (skip silent chunks)
    3. Debounce check                       < 1 ms  (prevent duplicate events)
    4. Assemble 0.96-second context window  < 1 ms  (ring buffer concatenation)
    5. YAMNet TFLite inference            ~100 ms  (Pi Zero 2W)
    6. Threshold decision                   < 1 ms
    7. InfluxDB write + CSV backup         ~50 ms   (if bark detected)
    ──────────────────────────────────────────────
    Total worst case                      ~353 ms  ✅ well under 1 second

Shutdown:
    - Ctrl+C (KeyboardInterrupt) for interactive use.
    - SIGTERM from systemd triggers a graceful stop via signal handler.

Usage:
    # Activate the virtual environment first:
    source ~/bark_env/bin/activate

    # Run directly:
    python bark_detector.py

    # Or via systemd:
    sudo systemctl start bark-detector
"""

import json
import logging
import os
import signal
import time
import wave
from datetime import datetime

import numpy as np

from audio_pipeline import AudioPipeline
from classifier import BarkClassifier
from influx_logger import InfluxLogger
from redis_config import RedisConfigWatcher

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

# Ensure the logs directory exists before opening the log file
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    # Include timestamp, level, and logger name for easy triage in journalctl
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        # Persistent log file — rotated manually or via logrotate
        logging.FileHandler("logs/bark_detector.log"),
        # Mirror output to stdout so `journalctl -u bark-detector -f` shows it
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("BarkDetector")


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config/settings.json") -> dict:
    """
    Load and return the JSON configuration file.

    All tuneable parameters (audio device, VAD thresholds, classifier settings,
    InfluxDB credentials) are kept in a single JSON file so they can be edited
    without touching any Python source.

    Args:
        path (str): Path to the settings file, relative to the working directory.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the settings file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r") as f:
        config = json.load(f)

    logger.info("Configuration loaded from: %s", path)
    return config


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class BarkDetector:
    """
    Orchestrates the full bark-detection pipeline.

    Responsibilities:
        - Initialise and wire together all sub-components.
        - Run the main event loop (blocking).
        - Handle confirmed bark detections (logging, snippet saving).
        - Collect runtime statistics for monitoring.
        - Ensure graceful shutdown on stop signals.

    Args:
        config (dict): Full settings dict loaded from config/settings.json.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.classifier_cfg = config["classifier"]

        # ----------------------------------------------------------------
        # Sub-component initialisation
        # ----------------------------------------------------------------

        # Audio pipeline: handles PyAudio stream + VAD + ring buffer
        self.audio = AudioPipeline(config)

        # Classifier: loads the YAMNet TFLite model into memory once
        self.classifier = BarkClassifier(self.classifier_cfg["model_path"])

        # InfluxDB logger: writes events to cloud + local CSV backup
        self.influx = InfluxLogger(config)

        # Redis watcher: polls Upstash for live threshold overrides (optional)
        self.redis_watcher = RedisConfigWatcher(config)

        # ----------------------------------------------------------------
        # Loop state
        # ----------------------------------------------------------------

        # Set to False by stop() or a signal handler to break the main loop
        self._running = False

        # Monotonic time after which the next bark can be detected.
        # Debouncing prevents the same prolonged bark from generating hundreds
        # of database writes during a single barking episode.
        self._debounce_until: float = 0.0

        # Running count of detected barks in the current session (for logs)
        self._bark_count: int = 0

        # ----------------------------------------------------------------
        # Runtime statistics
        # ----------------------------------------------------------------

        # Counters accumulate over the lifetime of the process.
        # Logged periodically to help diagnose performance issues.
        self._stats: dict = {
            "chunks_processed": 0,   # total 200-ms chunks read
            "vad_triggered": 0,      # chunks that passed the VAD check
            "inferences_run": 0,     # YAMNet inference invocations
            "barks_detected": 0,     # events that exceeded the threshold
            "start_time": time.time(),
        }

    # ------------------------------------------------------------------
    # Snippet saving
    # ------------------------------------------------------------------

    def _save_snippet(self, audio: np.ndarray) -> None:
        """
        Save a bark audio window as a WAV file for future training data collection.

        Files are stored in ``snippets/`` with a microsecond-resolution timestamp
        in the filename so they sort chronologically and never collide.

        A ring buffer of ``max_snippets`` files is maintained by deleting the
        oldest file whenever the count exceeds the limit, preventing the SD card
        from filling up over time.

        Args:
            audio (np.ndarray): float32 audio array (0.96-second YAMNet window).
        """
        os.makedirs("snippets", exist_ok=True)

        # Build filename from the current time with microsecond precision
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_path = f"snippets/bark_{timestamp_str}.wav"

        # Write a standard mono 16-bit PCM WAV file
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)          # mono
            wf.setsampwidth(2)          # 16-bit = 2 bytes per sample
            wf.setframerate(self.cfg["audio"]["sample_rate"])
            # Convert float32 [-1, 1] back to int16 for WAV storage
            pcm_data = (audio * 32768.0).astype(np.int16)
            wf.writeframes(pcm_data.tobytes())

        logger.debug("Snippet saved: %s", wav_path)

        # Enforce the ring-buffer size limit
        existing = sorted(os.listdir("snippets"))  # sorted = oldest first
        max_snippets = self.cfg["logging"]["max_snippets"]
        while len(existing) > max_snippets:
            oldest = existing.pop(0)
            os.remove(f"snippets/{oldest}")
            logger.debug("Oldest snippet removed: %s", oldest)

    # ------------------------------------------------------------------
    # Bark event handler
    # ------------------------------------------------------------------

    def on_bark(self, confidence: float, rms: float, audio: np.ndarray) -> None:
        """
        Called once per confirmed bark event after all checks pass.

        Actions performed:
            1. Increment session counters.
            2. Log to console + file (human-readable).
            3. Write to InfluxDB (+ CSV backup).
            4. Save the audio snippet for future training data.

        Args:
            confidence (float)      : YAMNet max bark-class confidence [0, 1].
            rms        (float)      : Linear RMS of the triggering audio chunk.
            audio      (np.ndarray) : 0.96-second context window used for inference.
        """
        self._bark_count += 1
        self._stats["barks_detected"] += 1

        logger.info(
            "BARK #%d detected! confidence=%.3f, rms=%.4f",
            self._bark_count,
            confidence,
            rms,
        )

        # Write the event to InfluxDB and the local CSV fallback
        self.influx.log_bark(
            confidence=confidence,
            rms=rms,
            audio=audio,
            dog_id=self.cfg.get("logging", {}).get("dog_id", "neighbor_unknown"),
        )

        # Save a WAV snippet for manual review and model fine-tuning
        self._save_snippet(audio)

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the real-time bark detection loop.

        This method blocks until ``stop()`` is called, a SIGTERM is received,
        or the user presses Ctrl+C.  All cleanup is performed in the ``finally``
        block regardless of how the loop exits.
        """
        self._running = True

        # Start the optional Redis config watcher in a background daemon thread
        self.redis_watcher.start()

        # Open the PyAudio stream — must be called before read_chunk()
        self.audio.start()

        logger.info("=" * 60)
        logger.info("Bark Detector running")
        logger.info("  confidence_threshold : %.2f", self.classifier_cfg["confidence_threshold"])
        logger.info("  debounce_seconds     : %.1f s", self.classifier_cfg["debounce_seconds"])
        logger.info("  rms_threshold (VAD)  : %.4f", self.cfg["vad"]["rms_threshold"])
        logger.info("=" * 60)

        try:
            while self._running:
                # --------------------------------------------------------
                # Step 1: Read one audio chunk from the microphone (~200 ms)
                # --------------------------------------------------------
                chunk = self.audio.read_chunk()
                self._stats["chunks_processed"] += 1

                # Compute RMS once here so we can pass it to on_bark() without
                # recalculating it inside influx_logger
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                # --------------------------------------------------------
                # Step 2: Voice Activity Detection (< 1 ms)
                # Skip classification entirely if the environment is silent.
                # This is the single biggest CPU saver on a Pi Zero 2W.
                # --------------------------------------------------------
                if not self.audio.vad_check(chunk):
                    continue  # quiet — do not run inference this iteration

                self._stats["vad_triggered"] += 1

                # --------------------------------------------------------
                # Step 3: Debounce check
                # After a bark is detected, suppress further detections for
                # debounce_seconds to avoid flooding InfluxDB during a long
                # barking episode.
                # --------------------------------------------------------
                if time.monotonic() < self._debounce_until:
                    continue  # still inside the cooldown window

                # --------------------------------------------------------
                # Step 4: Assemble the 0.96-second context window (< 1 ms)
                # YAMNet requires exactly 15 360 samples; the ring buffer in
                # AudioPipeline handles the assembly with zero-padding at startup.
                # --------------------------------------------------------
                context_audio = self.audio.get_context_audio(duration_s=0.96)

                # --------------------------------------------------------
                # Step 5: YAMNet inference (~80–120 ms on Pi Zero 2W)
                # --------------------------------------------------------
                t_infer_start = time.monotonic()
                result = self.classifier.predict(context_audio)
                inference_ms = (time.monotonic() - t_infer_start) * 1000
                self._stats["inferences_run"] += 1

                # --------------------------------------------------------
                # Step 6: Threshold decision
                # Re-read the threshold from config each iteration so that a
                # Redis update (via RedisConfigWatcher) takes effect immediately.
                # --------------------------------------------------------
                confidence = result["bark_confidence"]
                threshold = self.classifier_cfg["confidence_threshold"]

                if confidence >= threshold:
                    # Confirmed bark event — handle and start debounce timer
                    self.on_bark(confidence, rms, context_audio)
                    self._debounce_until = (
                        time.monotonic() + self.classifier_cfg["debounce_seconds"]
                    )

                # --------------------------------------------------------
                # Step 7: Periodic statistics logging (every 100 inferences)
                # --------------------------------------------------------
                if self._stats["inferences_run"] % 100 == 0:
                    uptime_min = (time.time() - self._stats["start_time"]) / 60
                    logger.info(
                        "Stats — barks: %d, inferences: %d, "
                        "last_inference: %.0f ms, uptime: %.1f min",
                        self._stats["barks_detected"],
                        self._stats["inferences_run"],
                        inference_ms,
                        uptime_min,
                    )

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received — stopping.")

        finally:
            # Guarantee cleanup regardless of how the loop exits
            self.stop()

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """
        Stop all sub-components and log final statistics.

        Safe to call multiple times (idempotent) — subsequent calls are no-ops
        because _running is set to False on the first call.
        """
        if not self._running:
            return  # already stopped

        self._running = False

        # Close the audio stream first to stop blocking read_chunk() calls
        self.audio.stop()

        # Signal the Redis watcher thread to exit cleanly
        self.redis_watcher.stop()

        # Flush any pending InfluxDB writes and close the HTTP connection
        self.influx.close()

        uptime_min = (time.time() - self._stats["start_time"]) / 60
        logger.info(
            "Bark Detector stopped after %.1f min — "
            "%d barks detected across %d inferences.",
            uptime_min,
            self._stats["barks_detected"],
            self._stats["inferences_run"],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load configuration, create the detector, wire up signal handlers, and run.

    Signal handling:
        SIGTERM is sent by systemd when `systemctl stop bark-detector` is issued.
        We translate it into a clean detector.stop() call so the process exits
        without leaving InfluxDB connections open or WAV files half-written.
    """
    config = load_config()
    detector = BarkDetector(config)

    # Register a SIGTERM handler for systemd-initiated shutdowns.
    # The lambda captures `detector` from the enclosing scope.
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: detector.stop(),
    )

    # Block here until the detector finishes
    detector.run()


if __name__ == "__main__":
    main()
