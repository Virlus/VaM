"""Image compatibility with the former recorded data JPEG writer, without a recorded data file."""

import cv2


def legacy_jpeg_rgb(image, color_order="rgb"):
    """Return the RGB pixels decoded by the legacy JPEG-95 collection path."""
    if color_order == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_order != "bgr":
        raise ValueError(f"Unsupported image color order: {color_order}")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Cannot encode camera image as JPEG")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Cannot decode camera JPEG")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
