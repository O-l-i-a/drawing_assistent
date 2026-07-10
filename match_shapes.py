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

# Monoline letter templates: thin pen-stroke contours, same technique as
# template_a (trace each stroke, retracing back over it where needed so the
# path always returns to its own start point) -- not filled block silhouettes.


def _arc(cx, cy, rx, ry, deg0, deg1, n=12):
    t = np.linspace(np.radians(deg0), np.radians(deg1), n)
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]


# B: spine + two bumps, each bump's arc returns to the spine on its own.
_b_upper = _arc(15, 25, 48, 25, -90, 90)
_b_lower = _arc(15, 75, 52, 25, -90, 90)
template_b = np.array(
    [(15, 100), (15, 0)] + _b_upper[1:] + _b_lower[1:],
    dtype=np.float32,
)

# C: a single open arc, traced out and back so the path closes on itself
# without a stray chord across the opening.
_c_arc = _arc(50, 50, 45, 45, 40, 320)
template_c = np.array(_c_arc + list(reversed(_c_arc))[1:], dtype=np.float32)

# D: spine, then a single bulging arc back from bottom to top.
_d_arc = _arc(15, 50, 50, 50, 90, -90)
template_d = np.array([(15, 0), (15, 100)] + _d_arc[1:], dtype=np.float32)

# E: spine with three prongs, each retraced back to the spine.
template_e = np.array([
    (15, 100), (15, 0),
    (75, 0), (15, 0),
    (15, 42), (65, 42), (15, 42),
    (15, 100), (75, 100), (15, 100),
], dtype=np.float32)

# F: spine with two prongs (no bottom one); the closing segment coincides
# with the spine itself, so no extra retrace is needed.
template_f = np.array([
    (15, 0), (75, 0), (15, 0),
    (15, 42), (65, 42), (15, 42),
    (15, 100),
], dtype=np.float32)

# G: like C, with a small spur retraced into the opening.
_g_arc = _arc(50, 50, 45, 45, 40, 320)
template_g = np.array(
    [(55, 77)] + _g_arc + list(reversed(_g_arc))[1:] + [(55, 77)],
    dtype=np.float32,
)

# H: two spines and a crossbar, retraced back to the start.
template_h = np.array([
    (15, 0), (15, 100),
    (15, 45), (65, 45),
    (65, 0), (65, 100),
    (65, 45), (15, 45), (15, 0),
], dtype=np.float32)

#template_list = [template_circle]
template_list = [
    template_a, template_square, template_triangle, template_circle,
    template_b, template_c, template_d, template_e, template_f, template_g, template_h,
]
template_names = ["A", "squ", "tri", "cir", "B", "C", "D", "E", "F", "G", "H"]
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
