from collections import defaultdict

from py_files.preprocessing import preprocess_plate
from py_files.ocr import recognize_plate


def ensemble_ocr(plate, reader):
    """
    Run OCR using several preprocessing methods.
    """

    methods = [
        "baseline",
        "equalize",
        "clahe",
        "otsu",
        "adaptive",
        "clahe_otsu"
    ]

    predictions = []

    for method in methods:

        processed = preprocess_plate(
            plate,
            method=method,
            use_super_resolution=True
        )

        text, conf = recognize_plate(
            processed,
            reader
        )

        if text is not None:

            predictions.append({

                "text": text,
                "confidence": conf,
                "method": method

            })

    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return predictions

from collections import Counter


def vote_predictions(predictions):
    """
    Vote for the most frequent OCR prediction.
    If several predictions have the same frequency,
    choose the one with the highest confidence.
    """

    if len(predictions) == 0:
        return None

    counter = Counter(
        p["text"]
        for p in predictions
    )

    max_votes = max(counter.values())

    candidates = [

        text

        for text, count in counter.items()

        if count == max_votes

    ]

    if len(candidates) == 1:
        return candidates[0]

    # tie -> choose highest confidence
    for pred in predictions:

        if pred["text"] in candidates:
            return pred["text"]


def correct_prediction(text):
    """
    Simple OCR correction for the first characters.
    """

    if text is None:
        return None

    corrections = {
        "5": "S",
        "0": "O",
        "8": "B"
    }

    text = list(text)

    # only first two symbols
    for i in range(min(2, len(text))):

        text[i] = corrections.get(
            text[i],
            text[i]
        )

    return "".join(text)


def best_prediction(
        plate,
        reader
):
    """
    Complete OCR post-processing pipeline.
    """

    predictions = ensemble_ocr(
        plate,
        reader
    )

    if len(predictions) == 0:

        return {

            "prediction": None,
            "raw_prediction": None,
            "confidence": 0,
            "best_method": None,
            "predictions": []

        }

    voted = vote_predictions(
        predictions
    )

    corrected = correct_prediction(
        voted
    )

    best = predictions[0]

    return {

        "prediction": corrected,

        "raw_prediction": voted,

        "confidence": best["confidence"],

        "best_method": best["method"],

        "predictions": predictions

    }