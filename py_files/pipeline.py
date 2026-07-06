from py_files.detection import extract_plate
from py_files.preprocessing import preprocess_plate
from py_files.ocr import recognize_plate

def predict_license_plate(
    image_path,
    model,
    reader
):

    plates = extract_plate(
        image_path,
        model
    )

    if len(plates) == 0:
        return None, None

    plate = preprocess_plate(
        plates[0]
    )

    text, confidence = recognize_plate(
        plate,
        reader
    )

    return text, confidence