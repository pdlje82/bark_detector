"""
audio_pipeline.py — Audio Capture, Voice Activity Detection, and Feature Extraction

Responsibilities:
    1. Open and manage a PyAudio input stream from the Behringer UM2 USB interface.
    2. Read fixed-size audio chunks (default: 200 ms) in a tight loop.
    3. Run a lightweight RMS-based Voice Activity Detector (VAD) that suppresses
       classifier invocations during silence, saving ~80 % of CPU on a Pi Zero 2W.
    4. Maintain a ring buffer of recent audio so a full 0.96-second context window
       (the exact input length required by YAMNet) is always available.

Design constraints:
    - No external ML dependencies — only PyAudio, NumPy, and the standard library.
    - All operations must complete well under one chunk duration (200 ms) to avoid
      accumulating latency.
    - The class is intentionally not thread-safe; it is designed for single-threaded
      use inside the main detector loop.

Typical usage:
    pipeline = AudioPipeline(config)
    pipeline.start()
    try:
        while True:
            chunk = pipeline.read_chunk()          # blocks for chunk_ms
            if pipeline.vad_check(chunk):          # skip silence
                audio = pipeline.get_context_audio()
                # pass audio to BarkClassifier ...
    finally:
        pipeline.stop()
"""

import logging
from collections import deque

import numpy as np
import pyaudio

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Manages the real-time audio input stream and provides pre-processed audio
    windows ready for the YAMNet classifier.

    Args:
        config (dict): Full settings dict loaded from config/settings.json.
                       Reads the ``audio`` and ``vad`` sub-sections.
    """

    def __init__(self, config: dict):
        # Store the relevant sub-sections for quick access
        self.cfg = config["audio"]
        self.vad_cfg = config["vad"]

        # Derive the number of raw int16 samples per chunk from the duration in ms.
        # Example: 16000 Hz * 200 ms / 1000 = 3200 samples per chunk.
        self.sample_rate: int = self.cfg["sample_rate"]
        self.chunk_size: int = int(self.sample_rate * self.cfg["chunk_ms"] / 1000)

        # Ring buffer that holds the last 1 second of audio as a sequence of chunks.
        # maxlen ensures old chunks are discarded automatically when the buffer is full.
        # Capacity: sample_rate / chunk_size chunks = exactly 1 second of audio.
        self.audio_buffer: deque = deque(
            maxlen=int(self.sample_rate / self.chunk_size)
        )

        # PyAudio instances — created in start(), destroyed in stop()
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

        # Counter used by the hysteresis VAD to require N consecutive loud chunks
        # before declaring that activity is present (prevents single-spike triggers).
        self._vad_counter: int = 0

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Open the PyAudio input stream.

        Raises:
            OSError: If the requested device_index does not exist or cannot be opened.
        """
        # Initialise the PortAudio back-end
        self._pa = pyaudio.PyAudio()

        # Open a blocking input stream.
        # paInt16 = signed 16-bit PCM, matching the int16 format specified in settings.
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.cfg["channels"],
            rate=self.sample_rate,
            input=True,
            input_device_index=self.cfg["device_index"],  # UM2 is usually card 1
            frames_per_buffer=self.chunk_size,             # buffer exactly one chunk
        )

        logger.info(
            "Audio stream started: device_index=%d, %d Hz, chunk=%d ms (%d samples)",
            self.cfg["device_index"],
            self.sample_rate,
            self.cfg["chunk_ms"],
            self.chunk_size,
        )

    def stop(self) -> None:
        """Stop and close the audio stream, then terminate PortAudio."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("Audio stream closed.")

    # ------------------------------------------------------------------
    # Audio reading
    # ------------------------------------------------------------------

    def read_chunk(self) -> np.ndarray:
        """
        Read exactly one chunk from the input stream.

        This call blocks for approximately ``chunk_ms`` milliseconds while the
        hardware fills the buffer.

        Returns:
            np.ndarray: float32 array of shape (chunk_size,) with values in [-1, 1].
                        Normalisation divides by 32768 (max positive int16 value).
        """
        # Read raw bytes from PortAudio; exception_on_overflow=False silently drops
        # frames if the processing loop fell behind — preferable to crashing.
        raw_bytes = self._stream.read(self.chunk_size, exception_on_overflow=False)

        # Interpret bytes as signed 16-bit integers (little-endian, native order on Pi)
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

        # Convert to float32 and normalise to the range [-1.0, 1.0].
        # YAMNet and most signal-processing code expect normalised floating-point audio.
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Append the new chunk to the ring buffer for context assembly
        self.audio_buffer.append(audio_float32)

        return audio_float32

    # ------------------------------------------------------------------
    # Voice Activity Detection
    # ------------------------------------------------------------------

    def vad_check(self, audio: np.ndarray) -> bool:
        """
        Decide whether the current chunk contains meaningful audio activity.

        Algorithm:
            Compute the Root Mean Square (RMS) energy of the chunk.  RMS is
            a cheap proxy for loudness that runs in < 0.1 ms on a Pi Zero 2W.

            A hysteresis counter is incremented each time RMS exceeds the
            threshold and decremented (clamped at 0) when it falls below.
            Activity is declared only after ``min_chunks_above`` consecutive
            loud chunks, which filters out single transient spikes (e.g. a
            camera click or a brief tap on the table) that are unlikely to be
            a dog bark.

        Args:
            audio (np.ndarray): float32 chunk returned by ``read_chunk()``.

        Returns:
            bool: True if sufficient energy is present to warrant classification.
        """
        # RMS = sqrt( mean( x² ) ) — a single vectorised NumPy call, very fast
        rms = float(np.sqrt(np.mean(audio ** 2)))

        if rms > self.vad_cfg["rms_threshold"]:
            # Loud chunk: increment activity counter (no upper bound needed)
            self._vad_counter += 1
        else:
            # Quiet chunk: decrement, but never go below zero
            self._vad_counter = max(0, self._vad_counter - 1)

        # Return True only once the counter reaches the minimum consecutive-chunk
        # requirement, providing hysteresis against one-shot transients
        return self._vad_counter >= self.vad_cfg["min_chunks_above"]

    # ------------------------------------------------------------------
    # Context window assembly
    # ------------------------------------------------------------------

    def get_context_audio(self, duration_s: float = 0.96) -> np.ndarray:
        """
        Assemble a fixed-length audio window from the ring buffer.

        YAMNet requires exactly 0.96 seconds = 15 360 samples at 16 kHz as its
        input.  This method concatenates all buffered chunks and returns the
        last ``duration_s`` seconds.  If the buffer has not yet accumulated
        enough audio (e.g. at startup), the beginning is zero-padded.

        Args:
            duration_s (float): Desired window length in seconds. Default 0.96 s
                                 matches YAMNet's expected input length.

        Returns:
            np.ndarray: float32 array of shape (sample_rate * duration_s,).
        """
        # Calculate how many samples we need
        target_samples = int(self.sample_rate * duration_s)

        # Concatenate all chunks currently in the ring buffer into one array.
        # list() creates a snapshot so the deque can be modified concurrently
        # (safe here because we are single-threaded, but good practice).
        buffered = np.concatenate(list(self.audio_buffer))

        if len(buffered) >= target_samples:
            # Take the most recent target_samples from the end of the buffer
            return buffered[-target_samples:]
        else:
            # Buffer not yet full (happens at startup) — prepend zeros so the
            # returned array always has exactly target_samples elements.
            padding = np.zeros(target_samples - len(buffered), dtype=np.float32)
            return np.concatenate([padding, buffered])
