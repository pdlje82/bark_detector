"""
influx_logger.py — InfluxDB Cloud Event Logger

Overview:
    Every confirmed bark event is persisted as a time-series data point in
    InfluxDB Cloud (free tier, EU Frankfurt region).  The schema is designed for
    downstream Grafana dashboards and future ML fine-tuning workflows.

    Data written per bark event:
        Measurement : bark_event
        Tags        : dog_id, model_version, location
        Fields      : confidence, rms_db, band_0 … band_4 (spectral energy bands)
        Timestamp   : UTC millisecond precision

    Reliability strategy:
        - Primary path: asynchronous InfluxDB write dispatched to a background
          thread via a queue.  The main detection loop returns immediately after
          enqueuing, removing ~50 ms of network I/O from the critical path.
        - Fallback path: append to a local CSV file synchronously on the calling
          thread before enqueuing the InfluxDB write.  The CSV is therefore
          always written even if the background thread encounters a network error
          or the process is killed before the queue is drained.

    Async write design:
        A single daemon thread (``_writer_thread``) blocks on a ``queue.Queue``.
        ``log_bark()`` constructs the InfluxDB Point and enqueues it along with
        the pre-computed CSV row.  The writer thread dequeues items one at a time
        and sends them to InfluxDB.  A sentinel value (``None``) is enqueued by
        ``close()`` to signal the thread to drain remaining items and exit.

    Spectral feature extraction:
        Rather than computing full MFCCs (which require librosa and are slow),
        we divide the FFT magnitude spectrum into 5 equal-width energy bands.
        These "spectral bands" are cheap to compute (< 1 ms) and still capture
        enough timbral information to distinguish bark types in Grafana or for
        lightweight ML experiments.

Dependencies:
    influxdb-client>=1.36  (pip install influxdb-client)
    numpy, scipy           (already required by audio_pipeline.py)
"""

import csv
import datetime
import logging
import os
import queue
import threading

import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from scipy.fft import rfft

logger = logging.getLogger(__name__)

# Number of spectral energy bands computed from the FFT magnitude spectrum.
# More bands = more resolution but also more InfluxDB fields per event.
NUM_SPECTRAL_BANDS = 5

# Maximum number of pending InfluxDB writes allowed in the queue.
# If the network is consistently slow and the queue fills up, new items are
# dropped (with a warning) rather than blocking the main detection loop.
MAX_QUEUE_SIZE = 50


class InfluxLogger:
    """
    Logs bark events to InfluxDB Cloud with a local CSV fallback.

    InfluxDB writes are dispatched asynchronously to a background thread so
    that the ~50 ms network round-trip does not block the main detection loop.
    The CSV backup is always written synchronously on the calling thread first,
    guaranteeing that no event is lost even if the process is killed mid-write.

    Args:
        config (dict): Full settings dict from config/settings.json.
                       Reads ``influxdb`` and ``logging`` sub-sections.
    """

    def __init__(self, config: dict):
        self.cfg = config["influxdb"]
        self.log_cfg = config["logging"]
        self.bucket = self.cfg["bucket"]

        # Build the InfluxDB client.  The client itself does not open a TCP
        # connection here; the connection is established on the first write.
        self._client = InfluxDBClient(
            url=self.cfg["url"],
            token=self.cfg["token"],
            org=self.cfg["org"],
        )

        # SYNCHRONOUS write API used inside the background thread.
        # The thread owns this object exclusively, so no locking is needed.
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

        # Bounded queue that decouples log_bark() (fast, main thread) from the
        # actual network I/O (slow, background thread).
        # Each item is a pre-built InfluxDB Point ready to send.
        self._write_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

        # Background writer thread — daemon so it does not prevent process exit
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="influx-writer",
            daemon=True,
        )
        self._writer_thread.start()

        # Path for the local CSV backup file
        self._csv_path = "logs/bark_events.csv"
        self._init_csv()

        logger.info("InfluxDB logger initialised (async). Bucket: %s", self.bucket)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        """
        Ensure the logs directory and CSV header row exist.

        Idempotent — safe to call multiple times.  The header is only written
        when creating a new file to avoid duplicating it on restarts.
        """
        # Create the logs directory if it does not already exist
        os.makedirs("logs", exist_ok=True)

        if not os.path.exists(self._csv_path):
            # Write header row for a new CSV file
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",    # ISO 8601 UTC
                    "dog_id",       # tag value, e.g. "neighbor_unknown"
                    "confidence",   # YAMNet max bark class score [0, 1]
                    "rms_db",       # RMS energy in decibels
                    "band_0",       # spectral energy band 0 (low frequencies)
                    "band_1",
                    "band_2",
                    "band_3",
                    "band_4",       # spectral energy band 4 (high frequencies)
                ])
            logger.info("Created new CSV backup file: %s", self._csv_path)

    # ------------------------------------------------------------------
    # Spectral feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_spectral_bands(audio: np.ndarray, num_bands: int = NUM_SPECTRAL_BANDS) -> list[float]:
        """
        Divide the real FFT magnitude spectrum into equal-width energy bands.

        This is a lightweight alternative to MFCCs: no mel filterbanks, no DCT,
        no librosa dependency.  Each band is the mean magnitude in that frequency
        range, which is fast (< 1 ms on Pi Zero 2W for 1024-sample FFT).

        Args:
            audio     (np.ndarray): float32 audio samples (any length; only the
                                     first 1024 samples are used for speed).
            num_bands (int)       : Number of frequency bands to compute.

        Returns:
            list[float]: ``num_bands`` mean magnitudes, ordered low → high frequency.
        """
        # Limit FFT to 1024 samples for speed (covers ~64 ms at 16 kHz)
        fft_input = audio[:1024]

        # Compute the one-sided (real) FFT magnitude spectrum
        magnitude_spectrum = np.abs(rfft(fft_input))

        # Split the spectrum into equal-width bands and compute mean per band
        band_size = len(magnitude_spectrum) // num_bands
        bands = [
            float(np.mean(magnitude_spectrum[i * band_size : (i + 1) * band_size]))
            for i in range(num_bands)
        ]
        return bands

    # ------------------------------------------------------------------
    # Background writer loop
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """
        Background thread: dequeue InfluxDB Points and send them one by one.

        Blocks on the queue until an item arrives, then attempts a synchronous
        write to InfluxDB.  Receiving ``None`` (the sentinel) signals that
        close() has been called; the loop drains any remaining items and exits.
        """
        logger.debug("InfluxDB writer thread started.")

        while True:
            # Block until a Point (or the None sentinel) is available
            item = self._write_queue.get()

            # None is the shutdown sentinel enqueued by close()
            if item is None:
                logger.debug("InfluxDB writer thread received shutdown signal.")
                break

            point, confidence, rms_db = item  # unpack the queued tuple

            try:
                self._write_api.write(bucket=self.bucket, record=point)
                logger.debug(
                    "InfluxDB write OK — confidence=%.3f, rms_db=%.1f dB",
                    confidence,
                    rms_db,
                )
            except Exception as exc:
                # Network errors are non-fatal; the CSV already captured the event
                logger.warning("InfluxDB write failed (event saved locally): %s", exc)
            finally:
                # Always mark the task done so queue.join() can unblock in close()
                self._write_queue.task_done()

        logger.debug("InfluxDB writer thread exiting.")

    # ------------------------------------------------------------------
    # Public logging interface
    # ------------------------------------------------------------------

    def log_bark(
        self,
        confidence: float,
        rms: float,
        audio: np.ndarray,
        dog_id: str = "neighbor_unknown",
        model_version: str = "yamnet_v1",
    ) -> None:
        """
        Persist a single bark event to InfluxDB and the local CSV.

        The CSV write is performed synchronously on the calling thread before
        anything else, guaranteeing the event is on disk within microseconds.
        The InfluxDB write is then enqueued for the background thread and this
        method returns immediately — removing ~50 ms of network I/O from the
        main detection loop's critical path.

        Args:
            confidence    (float)      : Max bark-class confidence score from YAMNet.
            rms           (float)      : RMS energy of the triggering audio chunk.
            audio         (np.ndarray) : The 0.96-second context audio window used
                                         for classification (float32, 16 kHz).
            dog_id        (str)        : InfluxDB tag identifying the dog source.
            model_version (str)        : InfluxDB tag for the classifier version.
        """
        # Capture the UTC timestamp at the moment of detection (on the calling
        # thread) so the timestamp reflects when the bark happened, not when
        # the background thread eventually sends the write.
        timestamp = datetime.datetime.utcnow()

        # Convert linear RMS to decibels.  The +1e-9 epsilon prevents log10(0).
        rms_db = float(20.0 * np.log10(rms + 1e-9))

        # Compute spectral band features from the audio context window
        bands = self._compute_spectral_bands(audio)

        # ----------------------------------------------------------------
        # Step 1: Local CSV backup — synchronous, always written first
        # ----------------------------------------------------------------
        # This guarantees the event is persisted to disk even if the process
        # is killed before the background thread processes the queue item.
        if self.log_cfg["local_csv_backup"]:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.isoformat(),
                    dog_id,
                    f"{confidence:.4f}",
                    f"{rms_db:.2f}",
                    f"{bands[0]:.4f}",
                    f"{bands[1]:.4f}",
                    f"{bands[2]:.4f}",
                    f"{bands[3]:.4f}",
                    f"{bands[4]:.4f}",
                ])

        # ----------------------------------------------------------------
        # Step 2: Build the InfluxDB Point on the calling thread
        # ----------------------------------------------------------------
        # Point construction is pure CPU work (no I/O) and completes in < 1 ms,
        # so it is safe and cheap to do here before enqueuing.
        point = (
            Point("bark_event")
            # Tags are indexed strings — ideal for filtering/grouping in Flux
            .tag("dog_id", dog_id)
            .tag("model_version", model_version)
            .tag("location", "garden")
            # Fields are the numeric measurements stored per event
            .field("confidence", confidence)
            .field("rms_db", rms_db)
            .field("band_0", bands[0])
            .field("band_1", bands[1])
            .field("band_2", bands[2])
            .field("band_3", bands[3])
            .field("band_4", bands[4])
            # Millisecond precision is more than sufficient for bark events
            .time(timestamp, WritePrecision.MILLISECONDS)
        )

        # ----------------------------------------------------------------
        # Step 3: Enqueue for the background writer thread (non-blocking)
        # ----------------------------------------------------------------
        try:
            # block=False raises queue.Full immediately if the queue is at
            # capacity, rather than stalling the main loop.
            self._write_queue.put_nowait((point, confidence, rms_db))
        except queue.Full:
            # This only happens if the network is so slow that MAX_QUEUE_SIZE
            # events have piled up without being sent.  The CSV already has the
            # data, so dropping the InfluxDB point is acceptable.
            logger.warning(
                "InfluxDB write queue full (%d items) — dropping point. "
                "Event is still in the local CSV.",
                MAX_QUEUE_SIZE,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Drain the write queue, stop the background thread, and close the client.

        Blocks until all enqueued InfluxDB writes have been sent (or failed),
        so no events are silently dropped on shutdown.
        """
        logger.info(
            "InfluxDB logger closing — draining %d queued writes...",
            self._write_queue.qsize(),
        )

        # Enqueue the sentinel to signal the writer thread to stop after
        # processing all items currently in the queue
        self._write_queue.put(None)

        # Wait for the writer thread to finish; 30 s timeout prevents hanging
        # indefinitely if the network is completely unreachable on shutdown
        self._writer_thread.join(timeout=30)

        if self._writer_thread.is_alive():
            logger.warning("InfluxDB writer thread did not exit within 30 s.")

        self._client.close()
        logger.info("InfluxDB client closed.")
