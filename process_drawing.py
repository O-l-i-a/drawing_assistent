"""Dummy processing module for detected polygons.

This will be replaced by the user's shape-recognition pipeline later.
The function `process_polygons` receives a list of polygons (each an Nx2 ndarray)
and the original warped image. For now it simply returns an empty list and
logs basic info when run as a script.
"""
from typing import List
import numpy as np


def process_polygons(polygons: List[np.ndarray], image: np.ndarray) -> List[dict]:
    """Placeholder processor.

    Args:
        polygons: list of Nx2 numpy arrays describing polygon vertices.
        image: the original warped BGR image.

    Returns:
        A list of dict placeholders for detected shapes (empty for now).
    """
    # simple placeholder: return a summary list
    summary = []
    for p in polygons:
        summary.append({"vertices": int(p.shape[0]), "area": float(np.abs(np.linalg.det(np.vstack([p, p[0]])))/2) if p.shape[0] >= 3 else 0.0})
    return summary


if __name__ == "__main__":
    print("process_drawing module loaded. Implement detection here.")
