from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image as RosImage


def ros_image_to_rgb_array(message: RosImage) -> np.ndarray:
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    height = message.height
    width = message.width
    encoding = message.encoding.lower()

    if encoding in {"rgb8", "bgr8"}:
        image = buffer.reshape(height, message.step)[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return image.copy()

    if encoding in {"rgba8", "bgra8"}:
        image = buffer.reshape(height, message.step)[:, : width * 4].reshape(height, width, 4)
        if encoding == "bgra8":
            image = image[:, :, [2, 1, 0, 3]]
        return image[:, :, :3].copy()

    if encoding in {"mono8", "8uc1"}:
        image = buffer.reshape(height, message.step)[:, :width]
        return np.repeat(image[:, :, None], 3, axis=2).copy()

    raise ValueError(f"Unsupported image encoding: {message.encoding}")
