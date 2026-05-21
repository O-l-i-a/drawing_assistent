try:
    from .app import main
    from .get_wrapped_paper import PaperDetectionConfig, PaperDetectionResult, get_wrapped_paper
    from .preprocess_drawing import preprocess_drawing
except ImportError:
    from app import main
    from get_wrapped_paper import PaperDetectionConfig, PaperDetectionResult, get_wrapped_paper
    from preprocess_drawing import preprocess_drawing


if __name__ == "__main__":
    main()
