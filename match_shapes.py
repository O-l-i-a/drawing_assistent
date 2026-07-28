import cv2 as cv
import numpy as np
from dataclasses import dataclass
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF

@dataclass 
class TemplateState:
    step_idx: int = 0
    box: object = None
    item: object = None
    name: str = ""
    view: object = None
    stable_count: int = 0

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

a_1 = np.array([
    [10, 100],
    [30,50],
    [50, 0],
], dtype=np.float32)
a_2 = np.array([
    [10, 100],
    [30,50],
    [50, 0],
    [70,50],
    [90, 100],
    [50, 0],
    [10, 100]
], dtype=np.float32)
a_3 = np.array([
    [10, 100],
    [30,50],
    [50, 0],
    [70,50],
    [90, 100],
    [70,50],
    [30,50],
    [10,100]
], dtype=np.float32)
a = [a_1,a_2,a_3]

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

triangle_1 = np.array([[0,100],[50,0]], dtype=np.float32)
triangle_2 = np.array([[0,100],[50,0],[100,100]], dtype=np.float32)
triangle_3 = np.array([[0,100],[50,0],[100,100],[0,100]], dtype=np.float32)
triangle = [triangle_1, triangle_2, triangle_3]


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
    a[-1], template_square, triangle[-1], template_circle,
    template_b, template_c, template_d, template_e, template_f, template_g, template_h,
]
line_by_line_template = [
    a, template_square, triangle, template_circle,
    template_b, template_c, template_d, template_e, template_f, template_g, template_h,
]

template_names = ["A", "squ", "tri", "cir", "B", "C", "D", "E", "F", "G", "H"]

template_map = dict(zip(template_names, line_by_line_template))

def template_to_canonical(step_pts, scale, x0, y0, item_pos=None, canonical_shape=None):
    pts = step_pts.astype(np.float32) * scale + np.array([x0, y0], dtype=np.float32)
    if item_pos is not None:
        pts_with_item = pts + np.array([item_pos.x(), item_pos.y()], dtype=np.float32)
        if canonical_shape is not None:
            h, w = canonical_shape[:2]
            if (pts_with_item.min() >= 0) and (pts_with_item.max() < max(w,h)):
                return pts_with_item
    return pts



def line_by_line(image, regions, state: TemplateState, x0, y0, scale):
    overlay = image.copy()  
    contour = state.item.scene_contour().astype(np.int32)

    template = template_map[state.name]

    full_template = template[-1].astype(np.float32)
    base_poly_qpoints = state.item._base_polygon
    base_pts = np.array([[p.x(), p.y()] for p in base_poly_qpoints], dtype=np.float32)

    M, inliers = cv.estimateAffinePartial2D(full_template, base_pts, method=cv.RANSAC, ransacReprojThreshold=2.0)

    step_pts = template_map[state.name][state.step_idx].astype(np.float32)

    if M is None:
        mins = full_template.min(axis=0)
        maxs = full_template.max(axis=0)
        extent = np.maximum(maxs - mins, 1e-6)
        canvas_rect = state.item._base_polygon.boundingRect()
        item_scale = canvas_rect.width() / extent.max()
        centered_scaled = (step_pts - mins - extent / 2.0) * item_scale
        poly = QPolygonF([QPointF(float(x), float(y)) for x, y in centered_scaled])
    else:
        ones = np.ones((step_pts.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([step_pts, ones])            # (N,3)
        transformed = (M @ pts_h.T).T                  # (N,2)
        poly = QPolygonF([QPointF(float(x), float(y)) for x, y in transformed])

    scene_poly = state.item.mapToScene(poly)

    contour_draw = contour.astype(np.int32).reshape((-1,1,2))
    step_contour = np.array([[p.x(), p.y()] for p in scene_poly], dtype=np.float32)
    step_contour = step_contour.reshape((-1,1,2)).astype(np.int32)

    cv.drawContours(overlay, [contour_draw], -1, (55, 55, 55), 5)
    cv.drawContours(overlay, [step_contour], -1, (0,255,0), 8)

    step_for_match = np.rint(step_contour).astype(np.int32).reshape(-1,2)
    sw_xmin, sw_ymin = step_for_match.min(axis=0)
    sw_xmax, sw_ymax = step_for_match.max(axis=0)
    sw_w = sw_xmax - sw_xmin
    tol_w = max(3, int(round(sw_w * 0.05)))

    if regions is not None:
        best_region = None
        best_score = float("inf")
       
        if len(regions) < 6: 
            step_for_match_simple = cv.approxPolyDP(step_contour, epsilon=1.0, closed=True)
            
            print("templ", sw_xmin,sw_ymin)
            for region in regions:
                
                rx, ry, rw, rh = region["bbox"]
                mask = region["mask"]
                inside = (
                    (rx >= sw_xmin - tol_w) &
                    (ry >= sw_ymin - tol_w) &
                    (rx + rw <= sw_xmax + tol_w) &
                    (ry + rh <= sw_ymax + tol_w)
                ).all()
                print("region", rx, ry, rw, rh)
                if not inside:
                    continue
                if rw < 3 or rh < 3:
                    continue


                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                region_contour = max(contours,  key=cv.contourArea)
                
                score = cv.matchShapes(region_contour, step_contour, cv.CONTOURS_MATCH_I1, 0)
                if score < best_score:
                        best_score = score
                        best_region = region
                print("score",score)
        
        if best_region is not None and best_score < 1:
            state.stable_count += 1
        else:
            state.stable_count = 0

        print("stable", state.stable_count)
        if state.stable_count > 5:
            if state.step_idx < len(template)-1:
                state.step_idx += 1
            state.stable_count = 0    
    
    output = cv.addWeighted(overlay, 0.5, image, 0.5, 0)
    return output

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
