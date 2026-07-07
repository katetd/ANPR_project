import cv2
from pathlib import Path

def to_gray(plate):
    return cv2.cvtColor(
        plate,
        cv2.COLOR_RGB2GRAY
    )


def resize(gray, scale=1):
    return cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


def equalize(gray):
    return cv2.equalizeHist(gray)


def clahe(gray):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    return clahe.apply(gray)


def gaussian(gray):
    return cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )


def otsu(gray):
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def adaptive(gray):
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
def preprocess_plate(
    plate,
    method="baseline",
    use_super_resolution=False
):
    if use_super_resolution:
        plate = super_resolve(plate)
        scale = 1
    else:
        scale = 6
    gray = to_gray(plate)
    gray = resize(
        gray,
        scale=scale
    )
    if method == "baseline":
        return gray
    elif method == "equalize":
        return equalize(gray)
    elif method == "clahe":
        return clahe(gray)
    elif method == "otsu":
        gray = gaussian(gray)
        return otsu(gray)
    elif method == "adaptive":
        gray = gaussian(gray)
        return adaptive(gray)

    elif method == "clahe_otsu":
        gray = clahe(gray)
        gray = gaussian(gray)
        return otsu(gray)

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


from pathlib import Path

sr = None


def load_super_resolution(model_path=None):

    global sr

    if sr is None:

        if model_path is None:

            model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "FSRCNN_x4.pb"
            )

        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        sr.readModel(str(model_path))

        sr.setModel("fsrcnn", 4)

    return sr

def super_resolve(image):

    sr = load_super_resolution()

    return sr.upsample(image)