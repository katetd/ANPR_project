def recognize_plate(image, reader):

    result = reader.readtext(
        image,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    if not result:
        return None, None
    result = sorted(
        result,
        key=lambda x: min(
            p[0]
            for p in x[0]
        )
    )
    text = "".join(
        r[1].upper()
        for r in result
    )
    confidence = sum(
        r[2]
        for r in result
    ) / len(result)
    return text, confidence