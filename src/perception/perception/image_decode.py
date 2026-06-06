"""Decodificacion robusta de sensor_msgs/Image a BGR (numpy).

Centraliza el fallback de encodings que usan los nodos de vision: intenta
`bgr8` directo y, si la camara publica otro encoding (rgb8, mono8, yuv...),
convierte manualmente. Devuelve None si no se pudo decodificar.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def imgmsg_to_bgr(bridge: CvBridge, msg: Image) -> Optional[np.ndarray]:
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception:
        pass
    try:
        raw = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    except Exception:
        return None
    enc = (msg.encoding or '').lower()
    if enc in ('rgb8', 'rgba8'):
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    if enc in ('mono8', '8uc1'):
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if enc.startswith('yuv'):
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
    if enc == 'bgra8':
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    return raw if raw is not None and raw.ndim == 3 else None
