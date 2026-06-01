import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

try:
    from .get_wrapped_paper import PaperDetectionConfig, get_wrapped_paper
    from .preprocess_drawing import preprocess_drawing
except ImportError:
    from get_wrapped_paper import PaperDetectionConfig, get_wrapped_paper
    from preprocess_drawing import preprocess_drawing


IMAGE_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "test1.jpeg"
DEFAULT_VIDEO_URL = "http://10.0.0.87:4747/video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a paper sheet, warp it, and optionally preprocess the drawing."
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
    last_corners: np.ndarray | None,
    args: argparse.Namespace,
    config: PaperDetectionConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    result = get_wrapped_paper(frame, last_corners, config)

    if args.preprocess and result.warped is not None:
        processed = preprocess_drawing(result.warped)
        display = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
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

    last_corners = None
    frame_count = 0

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

        display_image, last_corners = build_display_image(frame.copy(), last_corners, args, config)
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
