import argparse
import sys
from pathlib import Path

import cv2 as cv
from PySide6.QtWidgets import QApplication

try:
    from .gui.main_window import MainWindow
except ImportError:
    from gui.main_window import MainWindow

IMAGE_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "test1.jpeg"
DEFAULT_VIDEO_URL = "http://192.168.178.21:8080/video"


class StaticImageCapture:
    """Duck-types cv.VideoCapture so MainWindow's tick loop works with a still image."""

    def __init__(self, image_path: str):
        self._image = cv.imread(image_path)
        if self._image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

    def isOpened(self) -> bool:
        return True

    def set(self, *args, **kwargs) -> None:
        pass

    def read(self):
        return True, self._image.copy()

    def grab(self) -> None:
        pass

    def release(self) -> None:
        pass


def create_capture(source: str):
    if source.isdigit():
        return cv.VideoCapture(int(source))
    return cv.VideoCapture(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drawing assistant UI: place reference figures, draw them 1:1, and compare."
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
        help="Only process every Nth frame.",
    )
    parser.add_argument(
        "--orientation",
        choices=("portrait", "landscape"),
        default="portrait",
        help="Orientation of the physical paper, sets the shape of the paper canvas and "
        "live-feed panel. Frames are always fit without stretching regardless of this setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "image":
        capture = StaticImageCapture(args.image_path)
    else:
        capture = create_capture(args.video_source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {args.video_source}")

    app = QApplication(sys.argv)
    window = MainWindow(capture, frame_skip=args.frame_skip, orientation=args.orientation)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
