import cv2 as cv
import numpy as np


def preprocess_drawing(warped: np.ndarray) -> np.ndarray:
    """Dummy preprocessing placeholder for the warped paper image."""
    return cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
