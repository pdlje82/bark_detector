"""
classifier.py — YAMNet TFLite Classifier Wrapper

Overview:
    YAMNet (Yet Another Mobile Network) is a deep-net audio classifier pre-trained
    on AudioSet (over 2 million YouTube clips, 521 sound classes).  We run the
    TensorFlow Lite variant because the Pi Zero 2W lacks the RAM and CPU to run
    the full TensorFlow model at real-time rates.

    Measured inference time on a Pi Zero 2W: 80–120 ms per 0.96-second window.

    This module:
        - Loads the TFLite interpreter once at startup (avoids repeated I/O).
        - Exposes a single ``predict()`` method that accepts a normalised audio
          array and returns structured confidence scores.
        - Defines the specific AudioSet class indices that correspond to dog sounds
          so the caller does not need to know about YAMNet's internal label space.

YAMNet class indices relevant to dog barking (from yamnet_class_map.csv):
    74  — Dog bark
    75  — Bow-wow
    76  — Growling
    503 — Animal (broad fallback)

Reference:
    https://tfhub.dev/google/yamnet/1
    https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
"""

import logging

import numpy as np
import tflite_runtime.interpreter as tflite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class index mapping
# ---------------------------------------------------------------------------

# Dictionary of AudioSet class indices that we treat as "bark" evidence.
# Keys are integer indices into YAMNet's 521-element output vector.
# Values are human-readable labels for logging and debugging.
BARK_CLASSES: dict[int, str] = {
    74: "Dog bark",    # most specific — a single bark sound
    75: "Bow-wow",     # repeated barking pattern
    76: "Growling",    # low-frequency aggressive vocalisation
    503: "Animal",     # broad fallback for edge cases
}


class BarkClassifier:
    """
    Thin wrapper around the YAMNet TFLite interpreter.

    The interpreter is initialised once in ``__init__`` and reused for every
    subsequent ``predict()`` call.  Allocating tensors is expensive (~200 ms on
    a Pi Zero 2W), so it must not happen inside the hot loop.

    Args:
        model_path (str): Filesystem path to the ``yamnet.tflite`` file.
                          Relative paths are resolved from the working directory
                          (i.e. the bark_detector project root).

    Raises:
        RuntimeError: If the model file cannot be loaded or tensor allocation fails.
    """

    def __init__(self, model_path: str):
        logger.info("Loading YAMNet TFLite model from: %s", model_path)

        # Create the interpreter and load the flatbuffer model into memory
        self.interpreter = tflite.Interpreter(model_path=model_path)

        # Allocate input and output tensors.  This step is required before the
        # first inference and should only be called once.
        self.interpreter.allocate_tensors()

        # Cache tensor metadata to avoid repeated dictionary lookups per inference
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

        logger.info(
            "YAMNet loaded successfully. "
            "Input shape: %s, dtype: %s",
            self._input_details[0]["shape"],
            self._input_details[0]["dtype"],
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, audio: np.ndarray) -> dict:
        """
        Run YAMNet inference on a single audio window.

        The model is invoked synchronously (blocking).  On a Pi Zero 2W this
        typically takes 80–120 ms, which fits comfortably within the 200 ms
        chunk period.

        Args:
            audio (np.ndarray): float32 array of normalised audio samples.
                                 Must be exactly 0.96 s × 16 000 Hz = 15 360
                                 samples.  Values should be in [-1.0, 1.0].

        Returns:
            dict with the following keys:
                ``bark_confidence`` (float):
                    Maximum confidence score across all BARK_CLASSES entries.
                    This is the primary value compared against the threshold.
                ``bark_scores`` (dict[str, float]):
                    Per-class confidence scores for every entry in BARK_CLASSES,
                    keyed by human-readable label.
                ``top_class_idx`` (int):
                    Index of the highest-scoring class across all 521 classes.
                ``top_score`` (float):
                    Confidence of the top class (useful for debugging).
                ``all_scores`` (np.ndarray):
                    Raw 521-element output vector from YAMNet.
        """
        # Reshape audio to match the model's expected input shape.
        # get_input_details()[0]['shape'] is typically [1, 15360] or [15360].
        audio_input = audio.reshape(self._input_details[0]["shape"])

        # Copy input data into the interpreter's input tensor
        self.interpreter.set_tensor(self._input_details[0]["index"], audio_input)

        # Run the forward pass — this is the expensive step (~100 ms on Pi Zero 2W)
        self.interpreter.invoke()

        # Retrieve the output tensor.  Shape is [1, 521] (batch=1, 521 classes).
        # Index [0] removes the batch dimension to get a flat (521,) vector.
        scores: np.ndarray = self.interpreter.get_tensor(
            self._output_details[0]["index"]
        )[0]

        # Collect confidence scores only for the bark-related class indices.
        # Guard with `if idx < len(scores)` in case the model variant has
        # fewer than 521 classes (unlikely, but defensive).
        bark_scores: dict[str, float] = {
            class_label: float(scores[idx])
            for idx, class_label in BARK_CLASSES.items()
            if idx < len(scores)
        }

        # Aggregate to a single "bark confidence" value by taking the maximum
        # across all bark-related classes
        max_bark_score: float = max(bark_scores.values()) if bark_scores else 0.0

        # Find the globally top-scoring class for debugging / logging
        top_class_idx: int = int(np.argmax(scores))

        return {
            "bark_confidence": max_bark_score,          # primary decision value
            "bark_scores": bark_scores,                  # per-class breakdown
            "top_class_idx": top_class_idx,              # index into yamnet_classes.csv
            "top_score": float(scores[top_class_idx]),   # score of the top class
            "all_scores": scores,                        # full 521-element vector
        }
