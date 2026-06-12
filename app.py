import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

try:
    from .get_wrapped_paper import PaperDetectionConfig, PaperState, get_wrapped_paper, jpeg_to_paper_detection_result
    from .preprocess_drawing import preprocess_drawing
except ImportError:
    from get_wrapped_paper import PaperDetectionConfig, PaperState, get_wrapped_paper, jpeg_to_paper_detection_result
    from preprocess_drawing import preprocess_drawing


IMAGE_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "test1.jpeg"
DEFAULT_VIDEO_URL = "http://192.168.178.21:8080/video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a paper sheet, warp it, and optionally preprocess the drawing."
    )
    parser.add_argument(
        "--testing_wrapped",
        action="store_true",
        help="use the image as wrapped paper for testing the preprocess_drawing function.",
    )
    parser.add_argument(
        "--source",
        choices=("image", "video"),
        default="video",
        help="Choose whether to process a local image or a video stream.",
    )
    parser.add_argument(
        "--image-path",
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to the image file when using --source image.",
    )
    parser.add_argument(
        "--video-source",
        default=DEFAULT_VIDEO_URL,
        help="Camera URL or device index when using --source video.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=3,
        help="Only process every Nth frame in video mode.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the full detection collage.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Show the preprocessed warped drawing when a paper sheet is found.",
    )
    return parser.parse_args()


def create_capture(source: str) -> cv.VideoCapture:
    if source.isdigit():
        return cv.VideoCapture(int(source))
    return cv.VideoCapture(source)


def build_config(args: argparse.Namespace) -> PaperDetectionConfig:
    return PaperDetectionConfig(debug=args.debug)


def build_display_image(
    frame: np.ndarray,
    state: PaperState,
    args: argparse.Namespace,
    config: PaperDetectionConfig,
    validation_fail_count=0
) -> tuple[np.ndarray, np.ndarray | None]:
    result = get_wrapped_paper(frame, state, config)
    
    if args.preprocess and result.warped is not None:
        processed = preprocess_drawing(result.warped)

        # If testing with a wrapped image, override using CLI image_path
        if args.testing_wrapped:
            result = jpeg_to_paper_detection_result(args.image_path)
            processed = preprocess_drawing(result.warped)

        # support both old single-array return and new (vis, strokes) return
        if isinstance(processed, tuple):
            vis, strokes = processed
        else:
            vis = processed
            strokes = []

        # `vis` may be grayscale or already color (BGR). Handle both.
        if vis.ndim == 2:
            display = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        else:
            display = vis.copy()

        # create an overlay to draw masks/annotations in red, then blend once
        overlay = display.copy()
        used_overlay = False
        combined_mask = None

        # draw strokes/blobs/regions for debugging onto overlay
        for idx, stroke in enumerate(strokes):
            if stroke is None:
                continue

            # draw blob dicts as circles: {"center": (x,y), "size": float}
            if isinstance(stroke, dict) and "center" in stroke and "size" in stroke:
                center = (int(stroke["center"][0]), int(stroke["center"][1]))
                radius = max(1, int(stroke["size"] / 2))
                # draw shapes in solid red for clarity (B, G, R)
                color = (0, 0, 255)
                cv.circle(overlay, center, radius, color, 3, lineType=cv.LINE_AA)
                used_overlay = True
                continue

            # region dicts produced by region_growing: contain 'mask'
            if isinstance(stroke, dict) and "mask" in stroke:
                mask = stroke["mask"].astype(np.uint8)
                red = (0, 0, 255)
                overlay[mask == 255] = red
                x, y, w, h = stroke.get("bbox", (0, 0, 0, 0))
                cx, cy = stroke.get("centroid", (0, 0))
                cv.rectangle(overlay, (x, y), (x + w, y + h), red, 2)
                cv.circle(overlay, (cx, cy), 4, red, -1)
                used_overlay = True
                # accumulate mask to allow projection back onto camera frame
                if combined_mask is None:
                    combined_mask = mask.copy()
                else:
                    combined_mask = cv.bitwise_or(combined_mask, mask)
                continue

            # existing stroke/polygon drawing (dict with "points" or ndarray)
            pts = None
            if isinstance(stroke, dict):
                pts = stroke.get("points")
            elif isinstance(stroke, np.ndarray):
                pts = stroke
            if pts is None or pts.size == 0:
                continue
            pts = pts.astype(np.int32).reshape(-1, 1, 2)
            color = (int((37 * idx) % 256), int((79 * idx) % 256), int((151 * idx) % 256))
            cv.polylines(overlay, [pts], isClosed=False, color=color, thickness=2, lineType=cv.LINE_AA)

        # If we have a warped paper and corners, project combined mask back
        # onto the live overlay so regions appear in the camera view.
        if combined_mask is not None and result.warped is not None and result.corners is not None:
            h_warp, w_warp = result.warped.shape[:2]
            # destination coords in warped image space
            dst = np.array([[0, 0], [w_warp - 1, 0], [w_warp - 1, h_warp - 1], [0, h_warp - 1]], dtype=np.float32)
            src_corners = result.corners.astype(np.float32)
            # transform mask from warped -> frame overlay
            M = cv.getPerspectiveTransform(dst, src_corners)
            warped_back = cv.warpPerspective(combined_mask, M, (result.overlay.shape[1], result.overlay.shape[0]), flags=cv.INTER_NEAREST)

            frame_overlay = result.overlay.copy()
            red = (0, 0, 255)
            frame_overlay[warped_back == 255] = red

            # draw centroids projected back as well
            for stroke in strokes:
                if isinstance(stroke, dict) and "centroid" in stroke and "mask" in stroke:
                    cx, cy = stroke["centroid"]
                    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
                    pt_proj = cv.perspectiveTransform(pt, M)[0][0]
                    cv.circle(frame_overlay, (int(pt_proj[0]), int(pt_proj[1])), 4, red, -1)

            display = cv.addWeighted(frame_overlay, 0.85, result.overlay, 0.15, 0)
            return display, result.corners

        # blend overlay (red masks) with display if any overlay drawing happened
        if used_overlay:
            display = cv.addWeighted(overlay, 0.85, display, 0.15, 0)

        return display, result.corners

    if args.debug and result.collage is not None:
        return result.collage, result.corners

    return result.overlay, result.corners


def run_image_mode(args: argparse.Namespace, config: PaperDetectionConfig) -> None:
    image = cv.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image_path}")

    display_image, _ = build_display_image(image, None, args, config)

    while True:
        cv.imshow("Drawing Assistant", display_image)
        key = cv.waitKey(1)
        if key in (27, ord("q")):
            break

    cv.destroyAllWindows()


def run_video_mode(args: argparse.Namespace, config: PaperDetectionConfig) -> None:
    cap = create_capture(args.video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.video_source}")

    state = PaperState()
    frame_count = 0
    validation_fail_count = 0

    while True:
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.grab()
        if not ret:
            if cv.waitKey(1) == 27:
                break
            continue

        frame_count += 1

        if frame_count % max(1, args.frame_skip) != 0:
            cv.waitKey(1)
            continue

        display_image, last_corners = build_display_image(frame.copy(), state, args, config,validation_fail_count)
        state.last_corners = last_corners
        cv.imshow("Drawing Assistant", display_image)
        if cv.waitKey(1) in (27, ord("q")):
            break

    cap.release()
    cv.destroyAllWindows()


def main() -> None:
    args = parse_args()
    config = build_config(args)

     # Warm-up KMeans 
    dummy = np.array([[0], [1]], dtype=np.float32)
    KMeans(n_clusters=2, n_init=1).fit_predict(dummy)

    if args.source == "image":
        run_image_mode(args, config)
        return

    run_video_mode(args, config)
