from .visualizers import cv2_visualizer, matplotlib_visualizer

gui_visualizers = {
    "matplotlib": matplotlib_visualizer,
    "cv2": cv2_visualizer
}


def create_visualizer(visualizer: str) -> object:
    if visualizer not in gui_visualizers:
        raise KeyError(f"{visualizer} not valid "
                       f"({list(gui_visualizers.keys())})")
    return gui_visualizers[visualizer]()
