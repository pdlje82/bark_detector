"""
redis_config.py — Real-Time Configuration via Upstash Redis

Overview:
    This module allows threshold and sensitivity parameters to be changed at
    runtime without restarting the service.  It polls an Upstash Redis instance
    (TLS, serverless Redis-as-a-service) and merges any keys it finds into the
    in-memory config dictionary that the main detector loop reads.

    Use case example:
        While the detector is running on the Pi, you change the confidence
        threshold on your phone by setting a key in Upstash's web console.
        The detector picks up the new value on the next poll cycle (default 60s)
        without any SSH or systemctl interaction.

    Redis key schema:
        bark:confidence_threshold  →  float   e.g. "0.85"
        bark:rms_threshold         →  float   e.g. "0.020"
        bark:debounce_seconds      →  float   e.g. "2.0"
        bark:dog_id                →  str     e.g. "rex_neighbor"

    This module is optional.  If Redis credentials are missing or the connection
    fails, the detector continues with the static settings.json values.

Dependencies:
    redis>=5.0  (pip install redis)
"""

import logging
import time

logger = logging.getLogger(__name__)

# How often (in seconds) the detector polls Redis for config changes.
# 60 seconds is a reasonable default — changes take effect within a minute.
DEFAULT_POLL_INTERVAL_S = 60

# Redis key prefix for all bark-detector settings
KEY_PREFIX = "bark:"

# Mapping from Redis key suffix → (config section, field name, type coercer)
# Add new tuneable parameters here without changing any other code.
REDIS_KEY_MAP: dict[str, tuple[str, str, type]] = {
    "confidence_threshold": ("classifier", "confidence_threshold", float),
    "rms_threshold":        ("vad",        "rms_threshold",        float),
    "debounce_seconds":     ("classifier", "debounce_seconds",     float),
    "dog_id":               ("logging",    "dog_id",               str),
}


class RedisConfigWatcher:
    """
    Polls Upstash Redis for config overrides and merges them into the live
    config dict that the detector reads each loop iteration.

    Args:
        config (dict): Full settings dict (mutated in-place when Redis keys change).
        poll_interval_s (int): Seconds between Redis polls. Default 60.

    Usage:
        watcher = RedisConfigWatcher(config)
        watcher.start()   # launch background polling thread
        # ... detector runs ...
        watcher.stop()
    """

    def __init__(self, config: dict, poll_interval_s: int = DEFAULT_POLL_INTERVAL_S):
        self._config = config          # reference to the shared config dict
        self._interval = poll_interval_s
        self._running = False

        # Attempt to connect to Redis; if credentials are absent, mark as disabled
        self._client = self._connect()

    def _connect(self):
        """
        Create a Redis client from the ``redis`` section of settings.json.

        Returns None (and logs a warning) if the section is missing, incomplete,
        or the connection attempt raises an exception.

        Returns:
            redis.Redis | None
        """
        try:
            import redis  # imported lazily — not required if module is unused

            redis_cfg = self._config.get("redis", {})
            url = redis_cfg.get("url", "")
            password = redis_cfg.get("password", "")

            if not url or url.startswith("rediss://eu1-xxx"):
                # Placeholder value from settings.json — Redis not configured
                logger.info(
                    "Redis URL not configured; real-time config updates disabled."
                )
                return None

            # ssl=True is required for Upstash TLS endpoints (rediss://)
            client = redis.from_url(
                url,
                password=password,
                ssl=True,
                decode_responses=True,    # return str instead of bytes
                socket_connect_timeout=5, # fail fast if the host is unreachable
            )

            # Verify connectivity with a lightweight PING
            client.ping()
            logger.info("Redis connected: %s", url)
            return client

        except Exception as exc:
            logger.warning("Redis connection failed; using static config: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the background polling thread.

        The thread is marked as a daemon so it does not prevent process exit
        when the main loop finishes.
        """
        if self._client is None:
            return  # Redis not available — polling would always fail

        import threading

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Redis config watcher started (interval=%ds).", self._interval)

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to exit."""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=self._interval + 2)

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """
        Continuously fetch Redis keys and merge them into the in-memory config.

        Runs in a daemon thread; the main loop's config reads automatically
        see updated values because Python dicts are updated in-place.
        """
        while self._running:
            self._fetch_and_merge()
            # Sleep in 1-second increments so stop() responds quickly
            for _ in range(self._interval):
                if not self._running:
                    break
                time.sleep(1)

    def _fetch_and_merge(self) -> None:
        """
        Fetch all known bark keys from Redis and update the config dict.

        Errors during fetch are logged as warnings; the config is left unchanged,
        ensuring the detector always operates with the last-known-good values.
        """
        try:
            for key_suffix, (section, field, coerce) in REDIS_KEY_MAP.items():
                redis_key = KEY_PREFIX + key_suffix
                value_str = self._client.get(redis_key)

                if value_str is None:
                    continue  # Key not set in Redis — keep the current value

                # Convert the raw string to the correct Python type
                new_value = coerce(value_str)
                old_value = self._config[section].get(field)

                if new_value != old_value:
                    self._config[section][field] = new_value
                    logger.info(
                        "Config updated via Redis: %s.%s = %r (was %r)",
                        section, field, new_value, old_value,
                    )

        except Exception as exc:
            logger.warning("Redis poll error (config unchanged): %s", exc)
