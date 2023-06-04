from typing import Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf

from DatasetTools.utils.utils import ColorSource, RelativePosition


cfg = OmegaConf.create()


# ======= DATASET ==============================================================
cfg.DATASET = OmegaConf.create()
cfg.DATASET.PARSER: str = None
cfg.DATASET.META = OmegaConf.create()
cfg.DATASET.LABELS: Dict[int, str] = {}
cfg.DATASET.ANNOTATIONS_PATH: str = None
cfg.DATASET.IMAGES_PATH: str = None

# ======= VISUALIZATION PARAMETERS =============================================
# Images
cfg.VIS = OmegaConf.create()
cfg.VIS.IMG_WIDTH: Optional[int] = None
cfg.VIS.IMG_HEIGHT: Optional[int] = None

# Text
cfg.VIS.TEXT = OmegaConf.create()
cfg.VIS.TEXT.VISIBLE: bool = True
cfg.VIS.TEXT.SCALE: int = 1
cfg.VIS.TEXT.THICKNESS: int = 1
cfg.VIS.TEXT.COLOR: Tuple[int, int, int] = (255, 255, 255)
cfg.VIS.TEXT.ALPHA: float = 1
# Optional Seaborn color palette or list of colors
cfg.VIS.TEXT.PALETTE: Optional[Union[str, List[Tuple[int, int, int]]]] = None
cfg.VIS.TEXT.COLOR_SOURCE: ColorSource = ColorSource.LABEL
cfg.VIS.TEXT.LINE_SPACE: int = 15
cfg.VIS.TEXT.POSITION: RelativePosition = RelativePosition.BOTTOM_RIGHT

# Text background
cfg.VIS.TEXT_BG = OmegaConf.create()
cfg.VIS.TEXT_BG.VISIBLE: bool = True
cfg.VIS.TEXT_BG.COLOR: Tuple[int, int, int] = (224, 56, 84)
cfg.VIS.TEXT_BG.ALPHA: float = 0.75
# Optional Seaborn color palette or list of colors
cfg.VIS.TEXT_BG.PALETTE: Optional[Union[str, List[Tuple[int, int, int]]]] = None
cfg.VIS.TEXT_BG.COLOR_SOURCE: ColorSource = ColorSource.LABEL
cfg.VIS.TEXT_BG.MARGIN: int = 0

# Bounding boxes
cfg.VIS.BOX = OmegaConf.create()
cfg.VIS.BOX.VISIBLE: bool = True
cfg.VIS.BOX.THICKNESS: int = 1
cfg.VIS.BOX.COLOR: Tuple[int, int, int] = (224, 56, 84)
cfg.VIS.BOX.ALPHA: float = 1
# Optional Seaborn color palette or list of colors
cfg.VIS.BOX.PALETTE: Optional[Union[str, List[Tuple[int, int, int]]]] = None
cfg.VIS.BOX.COLOR_SOURCE: ColorSource = ColorSource.LABEL
cfg.VIS.BOX.TEXT_POSITION: RelativePosition = RelativePosition.TOP_LEFT
cfg.VIS.BOX.FILL: bool = False

# Segmentation masks
cfg.VIS.MASK = OmegaConf.create()
cfg.VIS.MASK.VISIBLE: bool = True
cfg.VIS.MASK.ALPHA: float = 0.75
cfg.VIS.MASK.COLOR: Tuple[int, int, int] = (21, 57, 255)
# Optional Seaborn color palette or list of colors
cfg.VIS.MASK.PALETTE: Optional[Union[str, List[Tuple[int, int, int]]]] = None
cfg.VIS.MASK.COLOR_SOURCE: ColorSource = ColorSource.LABEL

# Keypoints
cfg.VIS.KP = OmegaConf.create()
cfg.VIS.KP.VISIBLE: bool = True
cfg.VIS.KP.RADIUS: int = 5
cfg.VIS.KP.LINE_THICKNESS: int = 3
cfg.VIS.KP.NAMES_VISIBLE: bool = False
cfg.VIS.KP.CONNECTIONS_VISIBLE: bool = True
cfg.VIS.KP.KP_VISIBLE: bool = True
cfg.VIS.KP.COLOR: Tuple[int, int, int] = (0, 0, 255)
cfg.VIS.KP.COLOR_SOURCE: ColorSource = ColorSource.LABEL
cfg.VIS.KP.KP_COLORS: Dict[str, Tuple[int, int, int]] = {
    "left_ear": (255, 0, 255),
    "right_ear": (255, 255, 0),
    "left_eye": (223, 0, 255),
    "right_eye": (223, 255, 0),
    "nose": (255, 255, 255),
    "left_shoulder": (191, 0, 255),
    "right_shoulder": (191, 255, 0),
    "left_elbow": (159, 0, 255),
    "right_elbow": (159, 255, 0),
    "left_wrist": (127, 0, 255),
    "right_wrist": (127, 255, 0),
    "left_hip": (95, 0, 255),
    "right_hip": (95, 255, 0),
    "left_knee": (63, 0, 255),
    "right_knee": (63, 255, 0),
    "left_ankle": (31, 0, 255),
    "right_ankle": (31, 255, 0)
}
cfg.VIS.KP.CONNECTIONS: List[Tuple[str, str, Tuple[int, int, int]]] = [
    ("left_ear", "left_eye", (239, 0, 255)),
    ("right_ear", "right_eye", (239, 235, 0)),
    ("left_eye", "nose", (255, 0, 255)),
    ("nose", "right_eye", (255, 255, 0)),
    ("left_shoulder", "right_shoulder", (191, 255, 255)),
    ("left_shoulder", "left_elbow", (175, 0, 255)),
    ("right_shoulder", "right_elbow", (175, 255, 0)),
    ("left_elbow", "left_wrist", (143, 0, 255)),
    ("right_elbow", "right_wrist", (143, 255, 0)),
    ("left_hip", "right_hip", (95, 255, 255)),
    ("left_hip", "left_knee", (63, 0, 255)),
    ("right_hip", "right_knee", (63, 255, 0)),
    ("left_knee", "left_ankle", (47, 0, 255)),
    ("right_knee", "right_ankle", (47, 255, 0)),
    ("left_shoulder", "left_hip", (143, 0, 255)),
    ("right_shoulder", "right_hip", (143, 255, 0)),
    ("nose", "left_shoulder", (223, 0, 255)),
    ("nose", "right_shoulder", (223, 255, 0)),
]
