import cv2 as cv
import numpy as np

template_a = np.array([
    [10, 100],
    [30,50],
    [50, 0],
    [70,50],
    [90, 100],
    [70,50],
    [30,50],
    [10,100]
], dtype=np.float32)
template_square = np.array([
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100],
], dtype=np.float32)
template_triangle = np.array([
    [50, 0],
    [100, 100],
    [0, 100],  

], dtype=np.float32)
template_circle = np.array([
    [50 + 50 * np.cos(theta), 50 + 50 * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 100)
], dtype=np.float32)

#template_list = [template_circle]
template_list = [template_a, template_square, template_triangle, template_circle]
template_names = ["A","squ", "tri", "cir"]
def match_shapes(image, regions):
    regions_forms = []
    for region in regions:
        mask = region["mask"]
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = contours[0]
        best_shape = None
        best_score = 999
        best_name = None
        for i,template in enumerate(template_list):
            score = cv.matchShapes(contour, template, cv.CONTOURS_MATCH_I1, 0)
            print(template_names[i] + " " + str(score))
            if score < best_score:
                best_score = score
                best_shape = template
                best_name = template_names[i]
        regions_forms.append((region, best_shape, best_score,best_name))
        print("best: " + str(best_score))
    overlay = image.copy()
            
    for idx, (region, best_shape, best_score, name) in enumerate(regions_forms):
        if best_shape is None: continue
        if best_score > 1.0: continue
        x, y, w, h = region["bbox"]
        template_scaled = best_shape.copy().astype(np.float32)
        template_scaled[:, 0] *= w / 100.0
        template_scaled[:, 1] *= h / 100.0
        template_transformed = template_scaled + np.array([x, y], dtype=np.float32)
        template_contour = template_transformed.reshape((-1, 1, 2)).astype(np.int32)

        cv.drawContours(overlay, [template_contour], -1, (0, 255, 255), 3)
        cv.putText(overlay, f"{name} #{idx} ({best_score:.2f})", (x, max(0, y - 10)),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)

    output = cv.addWeighted(overlay, 0.5, image, 0.5, 0)
    return output
