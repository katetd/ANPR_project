import cv2


def to_gray(plate):
    return cv2.cvtColor(
        plate,
        cv2.COLOR_RGB2GRAY
    )


def resize(gray, scale=6):
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
    method="baseline"
):

    gray = to_gray(plate)
    gray = resize(gray)

    if method == "baseline":
        return gray

    elif method == "equalize":
        gray = equalize(gray)
        return gray

    elif method == "clahe":
        gray = clahe(gray)
        return gray

    elif method == "otsu":
        gray = gaussian(gray)
        gray = otsu(gray)
        return gray

    elif method == "adaptive":
        gray = gaussian(gray)
        gray = adaptive(gray)
        return gray

    elif method == "clahe_otsu":
        gray = clahe(gray)
        gray = gaussian(gray)
        gray = otsu(gray)
        return gray

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")