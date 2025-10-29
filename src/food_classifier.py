import os
import traceback
from functools import lru_cache
from threading import Lock
from typing import Optional, Tuple

import numpy as np


_LOAD_LOCK = Lock()
_MODEL = None
_LABELS = None
_LOAD_FAILED = False


def _default_model_paths() -> Tuple[str, str]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.getenv('FOOD_MODEL_PATH') or os.path.join(project_root, 'keras_model.h5')
    labels_path = os.getenv('FOOD_LABELS_PATH') or os.path.join(project_root, 'labels.txt')
    return model_path, labels_path


def _load_labels(labels_path: str):
    labels = []
    with open(labels_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            # Accept either "0 label" or just "label"
            parts = line.split()
            label = parts[-1]
            labels.append(label)
    return labels


def _ensure_model():
    global _MODEL, _LABELS
    if _MODEL is not None and _LABELS is not None:
        return

    with _LOAD_LOCK:
        if _MODEL is not None and _LABELS is not None:
            return

        model_path, labels_path = _default_model_paths()
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Keras model not found at {model_path}")
        if not os.path.exists(labels_path) or not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")

        try:
            from tensorflow.keras.models import load_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorFlow is required for food classification but is not installed. "
                "Install tensorflow>=2.12 to enable this feature."
            ) from exc

        _MODEL = load_model(model_path, compile=False)
        _LABELS = _load_labels(labels_path)


def _preprocess(frame: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    # Convert BGR (OpenCV) to RGB
    rgb = frame[:, :, ::-1]
    tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    tensor = tf.image.resize(tensor, (224, 224))
    tensor = tensor / 255.0
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor.numpy()


@lru_cache(maxsize=128)
def _label_safe(idx: int) -> Optional[str]:
    if _LABELS is None:
        return None
    if 0 <= idx < len(_LABELS):
        return _LABELS[idx]
    return None


def classify_food(frame: np.ndarray) -> Optional[Tuple[str, float]]:
    """
    Returns (label, confidence) if classification succeeds, otherwise None.
    """
    global _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    try:
        _ensure_model()
    except Exception:
        _LOAD_FAILED = True
        if os.getenv('FOOD_CLASSIFIER_DEBUG'):
            traceback.print_exc()
        return None

    if frame is None or frame.size == 0:
        return None

    try:
        batch = _preprocess(frame)
        predictions = _MODEL.predict(batch, verbose=0)
    except Exception:
        if os.getenv('FOOD_CLASSIFIER_DEBUG'):
            traceback.print_exc()
        return None

    if predictions.size == 0:
        return None

    scores = predictions[0]
    idx = int(np.argmax(scores))
    confidence = float(scores[idx])
    label = _label_safe(idx)
    if label is None:
        return None
    return label, confidence
