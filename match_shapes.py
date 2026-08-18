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
    transform: object = None  # 2x3 np.ndarray mapping template coords -> canvas/scene pixel coords
    adjusted: bool = False  # whether the one-time fit-to-drawn-ink correction has already happened
    complete: bool = False  # whether every step has been drawn and confirmed with "Next line"
    last_ink_mask: object = None  # snapshot of ink present at the last checkpoint (Start or Next line)
    manual_trigger: bool = False  # set by the "Next line" button, consumed on the following tick

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

# Which affine degrees of freedom "Next line" is allowed to correct for each
# figure when fitting the template to what was actually drawn. Symmetric
# reference shapes stay rigid; letters allow shear/anisotropic scale so
# slanted or narrower/wider handwriting (e.g. across scripts) still fits.
DEFAULT_TRANSFORM_FLAGS = {
    "rotate": True,
    "scale_uniform": True,
    "scale_anisotropic": False,
    "shear": False,
}
_LETTER_TRANSFORM_FLAGS = {**DEFAULT_TRANSFORM_FLAGS, "scale_anisotropic": True, "shear": True}

template_transform_flags: dict[str, dict[str, bool]] = {
    # A is fit with only rotation + translation + uniform scaling -- no
    # shear/anisotropic scale.
    "A": DEFAULT_TRANSFORM_FLAGS,
    "squ": DEFAULT_TRANSFORM_FLAGS,
    "tri": {**DEFAULT_TRANSFORM_FLAGS, "shear": True},
    "cir": {"rotate": False, "scale_uniform": True, "scale_anisotropic": False, "shear": False},
    "B": _LETTER_TRANSFORM_FLAGS,
    "C": _LETTER_TRANSFORM_FLAGS,
    "D": _LETTER_TRANSFORM_FLAGS,
    "E": _LETTER_TRANSFORM_FLAGS,
    "F": _LETTER_TRANSFORM_FLAGS,
    "G": _LETTER_TRANSFORM_FLAGS,
    "H": _LETTER_TRANSFORM_FLAGS,
}


def get_template_steps(name: str) -> list[np.ndarray]:
    """Normalize a template entry into a list of steps (single-shape templates
    that aren't broken into a stroke-by-stroke sequence become one step)."""
    steps = template_map[name]
    return steps if isinstance(steps, list) else [steps]


def template_to_canonical(step_pts, scale, x0, y0, item_pos=None, canonical_shape=None):
    pts = step_pts.astype(np.float32) * scale + np.array([x0, y0], dtype=np.float32)
    if item_pos is not None:
        pts_with_item = pts + np.array([item_pos.x(), item_pos.y()], dtype=np.float32)
        if canonical_shape is not None:
            h, w = canonical_shape[:2]
            if (pts_with_item.min() >= 0) and (pts_with_item.max() < max(w,h)):
                return pts_with_item
    return pts



def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return (M[:, :2] @ pts.T).T + M[:, 2]


def decompose_affine(M: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Decompose a 2x3 affine matrix into translation, rotation, scale x/y and
    shear (M = R . Shear . Scale -- the inverse of `compose_affine`)."""
    a, b, tx = M[0]
    c, d, ty = M[1]
    sx = float(np.hypot(a, c))
    if sx < 1e-9:
        return float(tx), float(ty), 0.0, 1e-9, 1e-9, 0.0
    angle = float(np.degrees(np.arctan2(c, a)))
    det = a * d - b * c
    sy = det / sx
    shear = (a * b + c * d) / det if abs(det) > 1e-9 else 0.0
    return float(tx), float(ty), angle, sx, float(sy), float(shear)


def compose_affine(tx: float, ty: float, angle: float, sx: float, sy: float, shear: float) -> np.ndarray:
    theta = np.radians(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    Sh = np.array([[1.0, shear], [0.0, 1.0]])
    S = np.array([[sx, 0.0], [0.0, sy]])
    A = R @ Sh @ S
    return np.array([[A[0, 0], A[0, 1], tx], [A[1, 0], A[1, 1], ty]], dtype=np.float64)


def stroke_centerline(points: np.ndarray, n: int = 24) -> np.ndarray:
    """Reduce a drawn stroke's raw (filled-width) pixel cloud to an n-point
    centerline ordered along its principal axis.

    Ink pixels come back in row-major scan order, several pixels wide (the
    stroke's thickness), not as a 1px-wide path. Sorting them by projection
    onto the principal axis and treating consecutive pixels as path steps
    (as `resample_polyline` does) massively inflates the apparent arc length
    -- every sideways jump across the stroke's width between two similarly-
    projected pixels adds a spurious step -- corrupting the fit. Bucketing
    into n bins along the axis and averaging each bin collapses the width
    away first, giving a clean centerline the same way `resample_polyline`
    can safely resample an actual polyline.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((n, 2))
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    if np.isscalar(cov) or cov.shape != (2, 2):
        return np.repeat(pts[:1], n, axis=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ major
    order = np.argsort(proj)
    proj_sorted = proj[order]
    pts_sorted = pts[order]

    span = proj_sorted[-1] - proj_sorted[0]
    if span < 1e-6:
        return np.repeat(pts_sorted[:1], n, axis=0)
    bin_edges = np.linspace(proj_sorted[0], proj_sorted[-1], n + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, proj_sorted, side="right") - 1, 0, n - 1)

    centerline = np.empty((n, 2))
    last_valid = pts_sorted[0]
    for i in range(n):
        sel = pts_sorted[bin_idx == i]
        if len(sel) > 0:
            centerline[i] = sel.mean(axis=0)
            last_valid = centerline[i]
        else:
            centerline[i] = last_valid
    return centerline


def _line_pose(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Centroid, principal-axis angle (deg, mod 180), length and width of a
    point cloud treated as a (possibly curved) line stroke."""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    if np.isscalar(cov) or cov.shape != (2, 2):
        return centroid, 0.0, 0.0, 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    minor = eigvecs[:, int(np.argmin(eigvals))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    proj_major = centered @ major
    length = float(proj_major.max() - proj_major.min())
    width = float(2.0 * np.std(centered @ minor))
    return centroid, angle, length, width


def union_mask(regions: list[dict], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for region in regions:
        region_mask = region.get("mask")
        if region_mask is not None and region_mask.shape == shape:
            mask = cv.bitwise_or(mask, region_mask)
    return mask


def region_mask_around_line(line_pts: np.ndarray, shape: tuple[int, int], radius: float = 200.0) -> np.ndarray:
    """A mask covering everywhere within `radius` pixels of the polyline
    `line_pts` -- the search window ink is looked for in, so a stroke drawn
    for a different step/figure elsewhere on the page can't get mistaken for
    the one just drawn for *this* line."""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.rint(np.asarray(line_pts, dtype=np.float64)).astype(np.int32)
    thickness = max(1, int(round(radius * 2)))
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 255, thickness, lineType=cv.LINE_8)
    elif len(pts) == 1:
        cv.circle(mask, tuple(pts[0]), int(round(radius)), 255, -1)
    return mask


def extract_new_ink(
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
    min_area: int = 15,
    search_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Pixels that are ink now but weren't in `previous_mask` -- the stroke(s)
    drawn since the last checkpoint. `search_mask`, if given, further limits
    this to a region of interest (see `region_mask_around_line`) so it uses
    the same ink threshold as everywhere else, just a restricted search
    area. Returns None if nothing new was found."""
    diff = cv.bitwise_and(current_mask, cv.bitwise_not(previous_mask))
    if search_mask is not None:
        diff = cv.bitwise_and(diff, search_mask)
    diff = cv.morphologyEx(diff, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(diff, connectivity=8)
    if num_labels <= 1:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    if stats[largest_label, cv.CC_STAT_AREA] < min_area:
        return None
    ys, xs = np.where(labels == largest_label)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def step_new_segment(steps: list[np.ndarray], step_idx: int) -> np.ndarray | None:
    """The portion of a step's template points that's new relative to the
    previous step, i.e. the one line the user is expected to add now.
    Returns None (never the whole shape) when steps aren't cleanly
    prefix-extending, so a fit is never mistakenly done against the entire
    figure instead of a single line."""
    current = np.asarray(steps[step_idx], dtype=np.float64)
    if step_idx == 0:
        return current
    prev = np.asarray(steps[step_idx - 1], dtype=np.float64)
    if len(prev) <= len(current) and np.allclose(current[: len(prev)], prev, atol=1e-3):
        # Include the last point of `prev` as the anchor, so this is a
        # proper line (from where the pen left off to the new point(s))
        # rather than just a lone new endpoint that can't define one.
        new_pts = current[len(prev) - 1:]
        if len(new_pts) >= 2:
            return new_pts
    return None


def resample_polyline(points: np.ndarray, n: int = 24) -> np.ndarray:
    """Resample a point set into n points evenly spaced along its arc length,
    so two differently-sampled strokes become directly comparable point-for-point."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((n, 2))
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2))
    for i, t in enumerate(targets):
        idx = int(np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1))
        span = seg_len[idx]
        frac = 0.0 if span < 1e-9 else (t - cum[idx]) / span
        out[i] = pts[idx] + frac * seg[idx]
    return out


def fit_step_transform(
    old_transform: np.ndarray,
    template_pts: np.ndarray,
    ink_pts: np.ndarray,
    flags: dict[str, bool],
    n_samples: int = 24,
    collinear_width_tol: float = 6.0,
) -> np.ndarray:
    """Fit a new template->scene affine transform so `template_pts` lines up
    with the ink actually drawn (`ink_pts`), constrained to only the degrees
    of freedom this figure allows (see `template_transform_flags`). Disabled
    degrees of freedom keep their value from `old_transform`."""
    if template_pts is None or ink_pts is None or len(template_pts) < 2 or len(ink_pts) < 5:
        return old_transform

    # ink_pts are raw, filled-width pixel coordinates, not a 1px path --
    # collapse them to a centerline first (see `stroke_centerline`) rather
    # than resampling the raw cloud, which would blow up the apparent arc
    # length with every sideways jump across the stroke's width.
    ink_centerline = stroke_centerline(ink_pts, n_samples)

    _, _, old_angle, old_sx, old_sy, old_shear = decompose_affine(old_transform)
    tmpl_centroid, tmpl_angle, tmpl_length, tmpl_width = _line_pose(template_pts)

    if tmpl_width < collinear_width_tol:
        # A (near-)straight template segment can't inform shear or
        # anisotropic scale -- infinitely many affines map one line onto
        # another equally well, so cv.estimateAffine2D's general 6-DOF
        # solve is degenerate here and returns an arbitrary, often wildly
        # wrong, matrix. Only rotation + uniform scale + translation are
        # actually well-posed from a single line; fit just those directly.
        ink_centroid, ink_angle, ink_length, _ = _line_pose(ink_centerline)

        # tmpl_angle/tmpl_length are measured on the raw, untransformed
        # template segment, so (ink_angle - tmpl_angle) and
        # (ink_length / tmpl_length) are already the absolute new
        # angle/scale -- not deltas to compose onto old_angle/old_sx, which
        # would double-apply whatever rotation/scale was already there.
        angle = old_angle
        if flags.get("rotate", True):
            # A line's angle is only defined mod 180 (no inherent
            # direction), so pick whichever full-angle candidate is closest
            # to old_angle instead of risking a spurious 180 degree flip.
            base = (ink_angle - tmpl_angle) % 180.0
            candidates = (base, base - 180.0, base + 180.0)
            angle = min(candidates, key=lambda a: abs(((a - old_angle + 180.0) % 360.0) - 180.0))

        sx, sy = old_sx, old_sy
        if (flags.get("scale_uniform", True) or flags.get("scale_anisotropic", False)) and tmpl_length > 1e-6:
            sx = sy = ink_length / tmpl_length

        rotated_scaled = compose_affine(0.0, 0.0, angle, sx, sy, old_shear)
        template_center_scene = apply_affine(rotated_scaled, tmpl_centroid[None, :])[0]
        tx, ty = ink_centroid - template_center_scene
        return compose_affine(tx, ty, angle, sx, sy, old_shear)

    tmpl_r = resample_polyline(template_pts, n_samples).astype(np.float32)
    ink_r = ink_centerline.astype(np.float32)

    candidates = []
    for ink_variant in (ink_r, ink_r[::-1]):
        M, _ = cv.estimateAffine2D(tmpl_r, ink_variant, method=cv.LMEDS)
        if M is not None:
            proj = (M[:, :2] @ tmpl_r.T).T + M[:, 2]
            residual = float(np.mean(np.linalg.norm(proj - ink_variant, axis=1)))
            candidates.append((residual, M))
    if not candidates:
        return old_transform
    _, best_M = min(candidates, key=lambda c: c[0])

    tx, ty, angle, sx, sy, shear = decompose_affine(best_M.astype(np.float64))

    if not flags.get("rotate", True):
        angle = old_angle
    if flags.get("scale_anisotropic", False):
        pass  # keep independent sx, sy from the fit
    elif flags.get("scale_uniform", True):
        sx = sy = (sx + sy) / 2.0
    else:
        sx, sy = old_sx, old_sy
    if not flags.get("shear", False):
        shear = old_shear

    return compose_affine(tx, ty, angle, sx, sy, shear)


def _diff_matches_expected(
    diff_pts: np.ndarray | None,
    expected_scene_pts: np.ndarray,
    angle_tol: float = 30.0,
    length_ratio_tol: float = 0.6,
) -> bool:
    """Cheap per-frame gate for automatic advance: does the ink drawn so far
    already look like the expected next stroke (similar direction and
    length)? Only used to decide when to auto-advance; "Next line" bypasses it."""
    if diff_pts is None or len(diff_pts) < 5 or expected_scene_pts is None or len(expected_scene_pts) < 2:
        return False
    _, diff_angle, diff_len, _ = _line_pose(diff_pts)
    _, exp_angle, exp_len, _ = _line_pose(expected_scene_pts)
    angle_diff = abs(diff_angle - exp_angle) % 180.0
    angle_diff = min(angle_diff, 180.0 - angle_diff)
    if angle_diff > angle_tol:
        return False
    if exp_len < 1e-3:
        return True
    ratio = diff_len / exp_len
    return (1 - length_ratio_tol) < ratio < (1 + length_ratio_tol)


def _initial_step_transform(state: "TemplateState") -> np.ndarray:
    """The starting template->scene transform, derived from how the user
    manually placed/rotated/resized the figure on the canvas before Start."""
    steps = get_template_steps(state.name)
    full_template = np.asarray(steps[-1], dtype=np.float32)
    base_poly_qpoints = state.item._base_polygon
    base_pts_local = np.array([[p.x(), p.y()] for p in base_poly_qpoints], dtype=np.float32)

    M_local, _ = cv.estimateAffinePartial2D(full_template, base_pts_local, method=cv.RANSAC, ransacReprojThreshold=2.0)
    if M_local is None:
        mins = full_template.min(axis=0)
        maxs = full_template.max(axis=0)
        extent = np.maximum(maxs - mins, 1e-6)
        rect = state.item._base_polygon.boundingRect()
        s = rect.width() / extent.max()
        M_local = np.array([
            [s, 0.0, -(mins[0] + extent[0] / 2.0) * s],
            [0.0, s, -(mins[1] + extent[1] / 2.0) * s],
        ], dtype=np.float64)

    local_pts = apply_affine(M_local.astype(np.float64), full_template)
    scene_poly = state.item.mapToScene(QPolygonF([QPointF(float(x), float(y)) for x, y in local_pts]))
    scene_pts = np.array([[p.x(), p.y()] for p in scene_poly], dtype=np.float32)

    M_scene, _ = cv.estimateAffinePartial2D(full_template, scene_pts, method=cv.RANSAC, ransacReprojThreshold=2.0)
    if M_scene is None:
        return M_local.astype(np.float64)
    return M_scene.astype(np.float64)


def line_by_line(image, regions, state: TemplateState, auto_advance: bool = True):
    """Draw the reference shape and the current step's target stroke over
    `image`, and advance once the user confirms a stroke was drawn.

    The figure's template->scene transform is fit to what was actually drawn
    exactly once: the first time a stroke is confirmed (manually via "Next
    line", `state.manual_trigger`, or -- if `auto_advance` is on --
    automatically after a few stable matching frames). From then on
    (`state.adjusted`) the transform is frozen; later "Next line" presses
    only advance to the next step, they never re-fit. Once the last step is
    confirmed, `state.complete` is set and this becomes a no-op that just
    keeps drawing the finished shape.
    """
    overlay = image.copy()
    steps = get_template_steps(state.name)

    if state.transform is None:
        state.transform = _initial_step_transform(state)

    full_template = np.asarray(steps[-1], dtype=np.float64)
    scene_full_pts = apply_affine(state.transform, full_template)
    contour_draw = np.rint(scene_full_pts).astype(np.int32).reshape(-1, 1, 2)

    if state.complete:
        # Nothing left to track -- just keep showing the finished shape.
        cv.drawContours(overlay, [contour_draw], -1, (55, 55, 55), 4)
        return cv.addWeighted(overlay, 0.5, image, 0.5, 0)

    step_idx = min(state.step_idx, len(steps) - 1)

    step_pts = np.asarray(steps[step_idx], dtype=np.float64)
    scene_step_pts = apply_affine(state.transform, step_pts)
    step_contour = np.rint(scene_step_pts).astype(np.int32).reshape(-1, 1, 2)
    # Green (current-step target) drawn first as a thick band, grey (full
    # reference shape) drawn second as a thinner line on top -- so where both
    # contours coincide (e.g. on the last step), the grey reference stays
    # visible as a line running through the green band instead of being
    # fully painted over by it.
    cv.drawContours(overlay, [step_contour], -1, (0, 255, 0), 10)
    cv.drawContours(overlay, [contour_draw], -1, (55, 55, 55), 4)

    # None when this step isn't a single, cleanly-isolated new line (see
    # `step_new_segment`) -- in that case we still advance below, we just
    # never fit against it, so a fit is always line-to-line, never shape-wide.
    new_segment_template = step_new_segment(steps, step_idx)
    scene_new_segment = apply_affine(state.transform, new_segment_template) if new_segment_template is not None else None

    current_mask = union_mask(regions, image.shape[:2]) if regions else np.zeros(image.shape[:2], dtype=np.uint8)
    # Ink is only looked for within a window around the expected line, using
    # the same ink threshold as everywhere else -- so a stroke drawn
    # elsewhere on the page (a different step, a different figure, stray
    # marks) can't get picked up as "the line just drawn" for this one.
    search_mask = region_mask_around_line(scene_new_segment, image.shape[:2]) if scene_new_segment is not None else None

    if state.last_ink_mask is None:
        state.last_ink_mask = current_mask
        diff_pts = None
    else:
        diff_pts = extract_new_ink(current_mask, state.last_ink_mask, search_mask=search_mask)

    if _diff_matches_expected(diff_pts, scene_new_segment):
        state.stable_count += 1
    else:
        state.stable_count = 0

    manual = state.manual_trigger
    state.manual_trigger = False
    should_advance = manual or (auto_advance and state.stable_count > 5)

    if should_advance:
        if not state.adjusted and new_segment_template is not None and diff_pts is not None and len(diff_pts) >= 5:
            flags = template_transform_flags.get(state.name, DEFAULT_TRANSFORM_FLAGS)
            state.transform = fit_step_transform(state.transform, new_segment_template, diff_pts, flags)
            state.adjusted = True

        state.stable_count = 0
        state.last_ink_mask = current_mask
        if step_idx < len(steps) - 1:
            state.step_idx = step_idx + 1
        else:
            state.complete = True

    return cv.addWeighted(overlay, 0.5, image, 0.5, 0)

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
