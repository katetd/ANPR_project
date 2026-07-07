from py_files.detection import extract_plate
from py_files.preprocessing import preprocess_plate
from py_files.ocr import recognize_plate

def predict_license_plate(
    image_path,
    model,
    reader,
    method="baseline"
):
    plates = extract_plate(
        image_path,
        model
    )
    if len(plates) == 0:
        return {
            "detected": False,
            "prediction": None,
            "confidence": 0,
            "plate": None
        }
    processed = preprocess_plate(
        plates[0],
        method=method
    )
    text, confidence = recognize_plate(
        processed,
        reader
    )
    return {
        "detected": True,
        "prediction": text,
        "confidence": confidence,
        "plate": processed
    }