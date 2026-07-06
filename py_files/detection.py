import cv2

from pathlib import Path
import cv2


def extract_plate(image_path, model):

    image_path = str(image_path)

    img = cv2.imread(image_path)

    if img is None:
        return []

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    results = model(image_path)

    plates = []

    for r in results:

        for box in r.boxes:

            xmin, ymin, xmax, ymax = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            plate = img[ymin:ymax, xmin:xmax]

            plates.append(plate)

    return plates