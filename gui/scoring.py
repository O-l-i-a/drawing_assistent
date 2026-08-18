import cv2 as cv
import numpy as np

try:
    from ..preprocess_drawing import preprocess_drawing
except ImportError:
    from preprocess_drawing import preprocess_drawing


def score_targets(
    canonical_bgr: np.ndarray,
    targets: list[dict],
    stroke_width: int = 6,
    tolerance: float = 25.0,
) -> tuple[list[dict], float]:
    """Compare placed target figures against what the user actually drew.

    Both the target and the drawn ink are thin strokes (monoline letters and
    outline shapes, never filled-in regions), so this scores them with a
    bidirectional distance-transform comparison rather than mask IoU:

    - `recall`: for every point of the target line, how close is the
      nearest ink? (did you draw the whole line?)
    - `precision`: for every drawn ink pixel, how close is the nearest
      target point? (did you stray off the line?)

    Each distance is converted to a 0-100 score via a smooth exponential
    falloff (`tolerance` = the distance, in pixels, at which the score is
    ~37%), and combined as their harmonic mean (F1-style) so a figure can't
    score well by either scribbling everywhere or drawing one tiny dot.

    All ink within a generous window around each target is aggregated
    first (not just the nearest single region), since a multi-stroke letter
    can easily end up as several disconnected regions.
    """
    h, w = canonical_bgr.shape[:2]
    _, regions = preprocess_drawing(canonical_bgr)

    ink_mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        ink_mask = cv.bitwise_or(ink_mask, region["mask"])

    results = []
    for target in targets:
        target_mask = np.zeros((h, w), dtype=np.uint8)
        target_pts = np.rint(target["points"]).astype(np.int32).reshape(-1, 1, 2)
        cv.polylines(target_mask, [target_pts], isClosed=True, color=255, thickness=stroke_width)

        # Only consider ink near this target, so a different figure's
        # strokes elsewhere on the page can't leak into this one's score.
        window = max(stroke_width * 4, 40)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (window, window))
        search_area = cv.dilate(target_mask, kernel)
        drawn_mask = cv.bitwise_and(ink_mask, search_area)

        if cv.countNonZero(drawn_mask) == 0:
            results.append({"name": target["name"], "score": 0.0, "precision": 0.0, "recall": 0.0, "matched": False})
            continue

        target_dist = cv.distanceTransform(cv.bitwise_not(target_mask), cv.DIST_L2, 5)
        drawn_dist = cv.distanceTransform(cv.bitwise_not(drawn_mask), cv.DIST_L2, 5)

        recall_dist = float(drawn_dist[target_mask == 255].mean())
        precision_dist = float(target_dist[drawn_mask == 255].mean())

        recall_score = 100.0 * np.exp(-recall_dist / tolerance)
        precision_score = 100.0 * np.exp(-precision_dist / tolerance)
        score = (
            2 * recall_score * precision_score / (recall_score + precision_score)
            if (recall_score + precision_score) > 0
            else 0.0
        )

        results.append({
            "name": target["name"],
            "score": score,
            "precision": precision_score,
            "recall": recall_score,
            "matched": True,
        })

    overall = float(np.mean([r["score"] for r in results])) if results else 0.0
    return results, overall
