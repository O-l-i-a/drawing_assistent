import cv2 as cv
import numpy as np

template_a = np.array([
    [10, 100],
    [50, 0],
    [90, 100],
    [70,50],
    [30,50],

], dtype=np.float32)


def match_shapes(image, regions):
    best_region = None
    best_score = 999
    for region in regions:
        mask = region["mask"]
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = contours[0]

        score = cv.matchShapes(contour,template_a,cv.CONTOURS_MATCH_I1, 0)
        if score < best_score:
            best_score = score
            best_region = region
    if best_region is None:
        return image

    x,y,w,h = best_region["bbox"]
    template_scaled = template_a.copy().astype(np.float32)
    template_scaled[:, 0] *= w / 100.0
    template_scaled[:, 1] *= h / 100.0
    
    template_transformed = template_scaled + np.array([x, y], dtype=np.float32)
    template_contour = template_transformed.reshape((-1, 1, 2)).astype(np.int32)
    
    overlay = image.copy()
    cv.drawContours(overlay, [template_contour], -1, (0,255,255), 3)
    output = cv.addWeighted(overlay, 0.5, image, 0.5, 0)
    print(output)
    print("[Match Shapes] Shape detected")
    return output
