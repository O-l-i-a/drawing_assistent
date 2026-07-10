import cv2 as cv
import numpy as np

try:
    from ..preprocess_drawing import preprocess_drawing
except ImportError:
    from preprocess_drawing import preprocess_drawing


def score_targets(canonical_bgr: np.ndarray, targets: list[dict]) -> tuple[list[dict], float]:
    """Compare placed target figures against what the user actually drew.

    Reuses `preprocess_drawing` (region growing) to segment the hand-drawn
    regions on the same canonical-sized paper image the targets were placed
    on, then matches each target to its nearest drawn region by centroid and
    scores the pair by mask IoU (position + shape + rotation + scale in one
    number), plus `cv.matchShapes` as a secondary detail metric.
    """
    h, w = canonical_bgr.shape[:2]
    _, regions = preprocess_drawing(canonical_bgr)

    available = list(regions)
    results = []

    for target in targets:
        target_mask = np.zeros((h, w), dtype=np.uint8)
        target_pts = target["points"].astype(np.int32).reshape(-1, 1, 2)
        cv.fillPoly(target_mask, [target_pts], 255)
        target_centroid = target["points"].mean(axis=0)

        best_idx = None
        best_dist = float("inf")
        for idx, region in enumerate(available):
            centroid = np.array(region["centroid"], dtype=np.float32)
            dist = float(np.linalg.norm(centroid - target_centroid))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None:
            results.append({"name": target["name"], "score": 0.0, "shape_distance": None, "matched": False})
            continue

        region = available.pop(best_idx)
        drawn_mask = region["mask"]

        intersection = cv.countNonZero(cv.bitwise_and(target_mask, drawn_mask))
        union = cv.countNonZero(cv.bitwise_or(target_mask, drawn_mask))
        iou = intersection / union if union else 0.0

        target_contours, _ = cv.findContours(target_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        drawn_contours, _ = cv.findContours(drawn_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        shape_distance = None
        if target_contours and drawn_contours:
            shape_distance = cv.matchShapes(target_contours[0], drawn_contours[0], cv.CONTOURS_MATCH_I1, 0)

        results.append({
            "name": target["name"],
            "score": iou * 100.0,
            "shape_distance": shape_distance,
            "matched": True,
        })

    overall = float(np.mean([r["score"] for r in results])) if results else 0.0
    return results, overall
