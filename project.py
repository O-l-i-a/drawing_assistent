import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


IMAGE_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "test1.jpeg"
DEFAULT_VIDEO_URL = "http://172.18.39.237:4747/video"


@dataclass
class AppConfig:
    source_type: str
    image_path: Path | None = None
    video_source: str | None = None
    crop_height: int = 80
    process_every_n_frames: int = 3
    window_name: str = "Drawing Assistant"


def callback(_input):
    pass


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Detect paper corners from either an image or a video stream."
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
        "--crop-height",
        type=int,
        default=80,
        help="Crop this many pixels from the top of each video frame.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=3,
        help="Only process every Nth frame in video mode.",
    )

    args = parser.parse_args()
    return AppConfig(
        source_type=args.source,
        image_path=Path(args.image_path),
        video_source=args.video_source,
        crop_height=args.crop_height,
        process_every_n_frames=max(1, args.frame_skip),
    )


def warm_up_kmeans() -> None:
    """Run a tiny dummy KMeans fit once to avoid first-use startup delay later."""
    dummy = np.array([[0], [1]], dtype=np.float32)
    KMeans(n_clusters=2, n_init=1).fit_predict(dummy)


def create_capture(source: str) -> cv.VideoCapture:
    """Create an OpenCV video capture object from a given source.

    Args:
        source: Video input source. If the value contains only digits,
            it is interpreted as a local camera index such as ``0``.
            Otherwise, it is treated as a stream URL or file path.

    Returns:
        cv.VideoCapture: An OpenCV capture object for the given source.
    """

    if source.isdigit():
        return cv.VideoCapture(int(source))
    return cv.VideoCapture(source)


def crop_frame(frame: np.ndarray, crop_height: int) -> np.ndarray:
    """Crop the top part of a video frame.

    Args:
        frame: Input image frame to crop.
        crop_height: Number of pixels to remove from the top of the frame.

    Returns:
        np.ndarray: The cropped frame. If ``crop_height`` is less than or
            equal to 0, the original frame is returned unchanged.
    """
    if crop_height <= 0:
        return frame

    height = frame.shape[0]
    crop_height = min(crop_height, height)
    return frame[crop_height:height, 0 : frame.shape[1]]


def process_frame(frame: np.ndarray, last_best_lines) -> tuple[np.ndarray | None, list | None]:
    """Run corner detection on a single frame.

    Args:
        frame: Input image frame to analyze.
        last_best_lines: Previously detected best lines, used as fallback
            if some lines cannot be detected in the current frame.

    Returns:
        tuple: A pair containing the output overlay image with detected
            lines drawn on it, and the updated list of best lines.
    """
    overlay, best_lines = corner_detection(frame, last_best_lines)
    return overlay, best_lines


def run_video_mode(config: AppConfig) -> None:
    """Process frames continuously from a video source.

    Args:
        config: Application configuration containing the video source,
            crop settings, frame sampling rate, and display options.

    Returns:
        None
    """
    cap = create_capture(config.video_source or DEFAULT_VIDEO_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.video_source}")

    best_lines = [None, None, None, None]
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
        frame = crop_frame(frame, config.crop_height)

        if frame_count % config.process_every_n_frames != 0:
            cv.waitKey(1)
            continue

        overlay, best_lines = process_frame(frame.copy(), best_lines)
        if overlay is None:
            continue

        cv.imshow(config.window_name, overlay)
        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def run_image_mode(config: AppConfig) -> None:
    """Load and process a single image, then display the result.

    Args:
        config: Application configuration containing the image path
            and display window settings.

    Returns:
        None
    """
    image_path = config.image_path or DEFAULT_IMAGE_PATH
    image = cv.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    overlay, _ = process_frame(image, [None, None, None, None])
    if overlay is None:
        raise RuntimeError("Corner detection did not return an image overlay.")

    while True:
        cv.imshow(config.window_name, overlay)
        key = cv.waitKey(1)
        if key in (27, ord("q")):
            break

    cv.destroyAllWindows()


def x_intersect(rho, theta) -> float:
    """Compute the x-axis intersection of a line in Hough form.

    Args:
        rho: Distance of the line from the origin in Hough space.
        theta: Angle of the line normal in radians.

    Returns:
        float: The x-coordinate where the line intersects the x-axis.
    """
    return rho / np.cos(theta)


def y_intersect(rho, theta) -> float:
    """Compute the y-axis intersection of a line in Hough form.

    Args:
        rho: Distance of the line from the origin in Hough space.
        theta: Angle of the line normal in radians.

    Returns:
        float: The y-coordinate where the line intersects the y-axis.
    """
    return rho / np.sin(theta)


def corner_detection(og_img, last_best_lines) -> tuple[np.ndarray, list]:
    """Detect the main paper boundary lines in an image.

    The function applies edge detection and a Hough transform to find
    horizontal and vertical line candidates, groups them with KMeans,
    selects the four most representative border lines, and draws them
    onto an output overlay.

    Args:
        og_img: Original input image in BGR format.
        last_best_lines: Previously detected boundary lines used as a
            fallback when some lines are missing in the current frame.

    Returns:
        tuple: A pair containing the overlay image with detected lines
            drawn on it, and the updated list of best boundary lines.
    """
    img = cv.cvtColor(og_img, cv.COLOR_BGR2RGB)
    overlay = og_img.copy()

    min_thresh = 30
    max_thresh = 90

    blur = cv.GaussianBlur(img, (5, 5), 0)
    canny_edge = cv.Canny(blur, min_thresh, max_thresh)
    edges = cv.dilate(canny_edge, None, iterations=1)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 80)

    colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    horizontal = []
    vertical = []
    vertical_abs_rho = []
    horizontal_abs_rho = []

    if lines is not None:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]

            abs_rho = rho
            if theta > np.pi / 2.0:
                abs_rho = abs(rho)

            is_vertical = (
                ((theta <= np.pi * 0.25 and theta >= 0) or (theta > np.pi * 1.75))
                or (theta > np.pi * 0.75 and theta <= np.pi * 1.25)
            )

            if is_vertical:
                vertical.append(line)
                vertical_abs_rho.append(abs_rho)
            else:
                horizontal.append(line)
                horizontal_abs_rho.append(abs_rho)

    if len(vertical) < 2 or len(horizontal) < 2:
        return overlay, last_best_lines

    vertical_rho = np.array(vertical_abs_rho, dtype=np.float32).reshape(-1, 1)
    horizontal_rho = np.array(horizontal_abs_rho, dtype=np.float32).reshape(-1, 1)
    if len(vertical_rho) == 0 or len(horizontal_rho) == 0:
        return overlay, last_best_lines

    try:
        vert_labels = KMeans(n_clusters=2, n_init=1).fit_predict(vertical_rho)
        hor_labels = KMeans(n_clusters=2, n_init=1).fit_predict(horizontal_rho)
    except Exception:
        return overlay, last_best_lines

    categorized_lines = [[], [], [], []]

    for index, label in enumerate(vert_labels):
        categorized_lines[label].append(vertical[index])

    for index, label in enumerate(hor_labels):
        categorized_lines[label + 2].append(horizontal[index])

    best_lines = [None, None, None, None]
    for index, category in enumerate(categorized_lines):
        if not category:
            continue

        mean_rho = sum(line[0][0] for line in category) / len(category)
        best_line = min(category, key=lambda line: abs(mean_rho - line[0][0]))

        rho = best_line[0][0]
        theta = best_line[0][1]

        if index < 2:
            xi = x_intersect(rho, theta)

            if best_lines[0] is None:
                best_lines[0] = best_line
            else:
                rho0 = best_lines[0][0][0]
                theta0 = best_lines[0][0][1]
                xi0 = x_intersect(rho0, theta0)

                if xi < xi0:
                    best_lines[1] = best_lines[0]
                    best_lines[0] = best_line
                else:
                    best_lines[1] = best_line
        else:
            yi = y_intersect(rho, theta)

            if best_lines[2] is None:
                best_lines[2] = best_line
            else:
                rho2 = best_lines[2][0][0]
                theta2 = best_lines[2][0][1]
                yi0 = y_intersect(rho2, theta2)

                if yi < yi0:
                    best_lines[3] = best_lines[2]
                    best_lines[2] = best_line
                else:
                    best_lines[3] = best_line

    if last_best_lines is not None:
        for index, line in enumerate(best_lines):
            if line is None and last_best_lines[index] is not None:
                best_lines[index] = last_best_lines[index]

    for index, line in enumerate(best_lines):
        if line is None:
            continue

        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        h, w = overlay.shape[:2]
        ok, clipped_pt1, clipped_pt2 = cv.clipLine((0, 0, w, h), pt1, pt2)

        if ok:
            cv.line(overlay, clipped_pt1, clipped_pt2, colors[index], 3, cv.LINE_AA)

    return overlay, best_lines


def main() -> None:
    """Run the application in image or video mode.

    Modes:
        image: Load a single image from disk and display the detected lines.
        video: Open a live video source and process frames continuously.

    Command-line arguments:
        --source: Select `image` or `video` mode.
        --image-path: Path to the input image for image mode.
        --video-source: URL, file path, or camera index for video mode.
        --crop-height: Number of pixels to crop from the top of each video frame.
        --frame-skip: Process every Nth frame in video mode.

    The function first warms up KMeans, then parses the arguments and
    starts the selected mode.
    """
    warm_up_kmeans()
    config = parse_args()

    if config.source_type == "image":
        run_image_mode(config)
        return

    run_video_mode(config)


if __name__ == "__main__":
    main()
