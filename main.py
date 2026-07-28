try:
    from .gui_app import main
except ImportError:
    from gui_app import main


if __name__ == "__main__":
    main()
