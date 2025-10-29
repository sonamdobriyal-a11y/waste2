import argparse
import os
import sys
from pathlib import Path

import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.food_classifier import classify_food  # noqa: E402


def load_image(path: Path):
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Unable to decode image at {path}")
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Test keras_model.h5 food classification on one or more images."
    )
    parser.add_argument(
        "images",
        metavar="IMAGE",
        nargs="+",
        help="Path(s) to the image file(s) to classify.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="Override path to keras_model.h5",
    )
    parser.add_argument(
        "--labels",
        dest="labels",
        help="Override path to labels.txt",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["FOOD_MODEL_PATH"] = args.model
    if args.labels:
        os.environ["FOOD_LABELS_PATH"] = args.labels

    for image_path in args.images:
        path = Path(image_path).resolve()
        try:
            frame = load_image(path)
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")
            continue

        result = classify_food(frame)
        if result is None:
            print(f"[WARN] {path}: classification failed or model unavailable.")
        else:
            label, confidence = result
            pct = confidence * 100
            print(f"[OK] {path}: {label} ({pct:.2f}%)")


if __name__ == "__main__":
    main()

