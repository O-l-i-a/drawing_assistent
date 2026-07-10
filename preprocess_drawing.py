import cv2 
import numpy as np
from collections import deque


def black_white_image(image: np.ndarray, thresh: int | None = None) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # optional local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # estimate smooth background and make strokes positive
    k = max(31, (min(g.shape) // 20) | 1)
    bg = cv2.medianBlur(g, k)
    diff = cv2.subtract(bg, g)  # dark strokes -> large positive values

    # normalize so the largest (darkest strokes) becomes 255
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv = cv2.bitwise_not(norm)
    # final binary image (choose Otsu/adaptive or fixed threshold)
    if thresh is None:
        _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(inv, thresh, 255, cv2.THRESH_BINARY)
    return bw

def preprocess_drawing(warped: np.ndarray) -> np.ndarray:
    """Preprocess the warped paper image and return a binary visual and regions.

    Uses the region-growing implementation below.
    """
    # keep a binary visual for downstream use
    vis = black_white_image(warped)

    # region_growing accepts either a path or an ndarray image
    regions = region_growing(warped, min_area=50)

    return vis, regions



def region_growing(image_or_path: str | np.ndarray, min_area: int = 100) -> list[dict]:
    """
    Segment black shapes via region growing.

    Accepts either a file path (str) or an image ndarray. Returns list of regions
    where each region is a dict: {id, mask, bbox, area, centroid}.
    """
    # Load image if a path was provided
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_or_path}")
    else:
        img = image_or_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # THRESH_BINARY_INV: black ink -> 255 (foreground), white bg -> 0
    binary = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31, 5)

    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)

    # 4-connected neighbours
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    regions = []

    # collect foreground coords for seed lookup
    foreground_ys, foreground_xs = np.where(binary == 255)

    for seed_y, seed_x in zip(foreground_ys, foreground_xs):
        if visited[seed_y, seed_x]:
            continue

        queue = deque()
        queue.append((seed_y, seed_x))
        visited[seed_y, seed_x] = True
        region_pixels = []

        while queue:
            cy, cx = queue.popleft()
            region_pixels.append((cy, cx))

            for dy, dx in DIRS:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if not visited[ny, nx] and binary[ny, nx] == 255:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        if len(region_pixels) < min_area:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        pixel_arr = np.array(region_pixels)
        mask[pixel_arr[:, 0], pixel_arr[:, 1]] = 255

        ys, xs = pixel_arr[:, 0], pixel_arr[:, 1]
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        centroid = (int(xs.mean()), int(ys.mean()))

        regions.append({
            "id": len(regions),
            "mask": mask,
            "bbox": bbox,
            "area": len(region_pixels),
            "centroid": centroid,
        })

    return regions


def visualize_regions(image: np.ndarray, regions: list[dict], out_path: str = "output_regions.png") -> None:
    """Overlay detected regions on `image` in solid red, draw bboxes and centroids, and save."""
    if image.ndim == 2:
        disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        disp = image.copy()

    overlay = disp.copy()
    red = (0, 0, 255)

    for r in regions:
        mask = r["mask"]
        # color overlay where mask is foreground
        overlay[mask == 255] = red
        x, y, w, h = r["bbox"]
        cx, cy = r["centroid"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), red, 2)
        cv2.circle(overlay, (cx, cy), 4, red, -1)

    # blend for semi-transparent effect
    out = cv2.addWeighted(overlay, 0.85, disp, 0.15, 0)
    cv2.imwrite(out_path, out)
    print(f"[Region Growing] Wrote visualization to {out_path}. Found {len(regions)} region(s).")