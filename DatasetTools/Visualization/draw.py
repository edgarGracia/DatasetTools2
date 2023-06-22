from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import seaborn as sns
from omegaconf import DictConfig
import seaborn as sns

from DatasetTools.Config.config import get_cfg, update_copy
from DatasetTools.utils.utils import ColorSource, RelativePosition
from DatasetTools.Structures.instance import Instance
from DatasetTools.Structures.image import Image
from DatasetTools.Structures import bounding_box

CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_color(
    color_source: ColorSource,
    color: Tuple[int, int, int],
    palette: Optional[Union[str, List[Tuple[int, int, int]]]],
    instance: Optional[Instance] = None
) -> Tuple[int, int, int]:
    """Get the color of an instance for the given parameters.

    Args:
        color_source (ColorSource): The color source.
        color (Tuple[int, int, int]): A single color.
        palette (Optional[Union[str, List[Tuple[int, int, int]]]]): The name of
            a Seaborn palette, list of colors or None.
        instance (Optional[Instance], optional): Optional instance used to
            determine the color.

    Raises:
        NotImplementedError

    Returns:
        Tuple[int, int, int]: A color.
    """
    if color_source is ColorSource.SOLID:
        return color
    if color_source is ColorSource.LABEL:
        if instance is None or instance.label_id is None:
            return color
        if isinstance(palette, str) or palette is None:
            return [
                int(i*255) for i in
                sns.color_palette(
                    palette, instance.label_id+1
                )[instance.label_id]
            ]
        return palette[instance.label_id]
    if color_source is ColorSource.INSTANCE:
        if instance is None or instance.instance_id is None:
            return color
        if isinstance(palette, str) or palette is None:
            return [
                int(i*255) for i in
                sns.color_palette(
                    palette, instance.instance_id+1
                )[instance.instance_id]
            ]
        return palette[instance]
    raise NotImplementedError


def draw_text(
    image: np.ndarray,
    text: Union[str, List[str]],
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
    thickness: int = 1,
    line_space: int = 15,
    relative_position: RelativePosition = RelativePosition.TOP_LEFT,
    background: bool = True,
    background_color: Tuple[int, int, int] = (50, 50, 50),
    background_alpha: float = 1,
    background_margin: int = 0
) -> np.ndarray:
    """Draw text in an image.

    TODO

    Raises:
        NotImplementedError

    Returns:
        np.ndarray: The image with the text drawn.
    """
    if not text:
        return image

    text_lines = text.splitlines() if isinstance(text, str) else text
    max_text_len = max(text_lines, key=lambda x: len(x))

    (text_w, text_h), _ = cv2.getTextSize(
        max_text_len,
        CV2_FONT,
        scale,
        thickness
    )

    if relative_position is RelativePosition.TOP_LEFT:
        x, y = position
    else:
        raise NotImplementedError

    if background and background_alpha > 0:
        box_h = (text_h + (scale * line_space)) * (len(text_lines)-1)
        xmin = max(0, x - background_margin)
        ymin = max(0, y - text_h - background_margin)
        xmax = min(image.shape[1], x + text_w + background_margin)
        ymax = min(image.shape[0], y + box_h + background_margin)

        if background_alpha < 1:
            bg = np.full(
                (ymax-ymin, xmax-xmin, 3),
                background_color,
                dtype="uint8"
            )
            image[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(
                image[ymin:ymax, xmin:xmax, :],
                1-background_alpha,
                bg,
                background_alpha,
                0
            )
        else:
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                background_alpha,
                -1
            )

    for i, line in enumerate(text_lines):
        dy = i * (text_h + scale * line_space)
        cv2.putText(
            image,
            line,
            (x, y+dy),
            CV2_FONT,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    return image


# def draw_text(
#     text: str,
#     image: np.ndarray,
#     position: Tuple[int, int],
#     instance: Optional[Instance] = None,
#     cfg: Optional[DictConfig] = None,
#     **kwargs
# ) -> np.ndarray:
#     """Draw a text in an image.

#     Args:
#         text (str): The text to draw.
#         image (np.ndarray): Numpy image of shape (H, W, 3) and ``uint8`` dtype.
#         position (Tuple[int, int]) Text position.
#         instance (Optional[Instance], optional): Optional instance used to
#             select the color. Default to None.
#         cfg (Optional[DictConfig], optional): A configuration object.
#             If None, the default will be used. Defaults to None.
#         **kwargs: cfg values to override.

#     Raises:
#         NotImplementedError

#     Returns:
#         np.ndarray: The image with the text drawn.
#     """
#     if not text:
#         return image

#     if cfg is None:
#         cfg = get_cfg()

#     if kwargs:
#         cfg = update_copy(cfg, kwargs)

#     lines = text.splitlines()
#     max_text_len = max(lines, key=lambda x: len(x))
#     (text_w, text_h), _ = cv2.getTextSize(
#         max_text_len,
#         FONT,
#         cfg.VIS.TEXT.SCALE,
#         cfg.VIS.TEXT.THICKNESS
#     )

#     if cfg.VIS.TEXT.POSITION is RelativePosition.TOP_RIGHT:
#         x, y = position
#     else:
#         raise NotImplementedError

#     if cfg.VIS.TEXT_BG.VISIBLE:
#         bg_color = get_color(
#             cfg.VIS.TEXT_BG.COLOR_SOURCE,
#             cfg.VIS.TEXT_BG.COLOR,
#             cfg.VIS.TEXT_BG.PALETTE,
#             instance
#         )

#         box_h = (text_h + (cfg.VIS.TEXT.SCALE *
#                  cfg.VIS.TEXT.LINE_SPACE)) * (len(lines)-1)

#         xmin = max(0, x - cfg.VIS.TEXT_BG.MARGIN)
#         ymin = max(0, y - text_h - cfg.VIS.TEXT_BG.MARGIN)
#         xmax = min(image.shape[1], x + text_w + cfg.VIS.TEXT_BG.MARGIN)
#         ymax = min(image.shape[0], y + box_h + cfg.VIS.TEXT_BG.MARGIN)

#         if cfg.VIS.TEXT_BG.ALPHA < 1:
#             bg = np.full((ymax-ymin, xmax-xmin, 3), bg_color, dtype="uint8")
#             image[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(
#                 image[ymin:ymax, xmin:xmax, :],
#                 1-cfg.VIS.TEXT_BG.ALPHA,
#                 bg,
#                 cfg.VIS.TEXT_BG.ALPHA,
#                 0
#             )
#         else:
#             cv2.rectangle(
#                 image,
#                 (xmin, ymin),
#                 (xmax, ymax),
#                 bg_color,
#                 -1
#             )

#     text_color = get_color(
#         cfg.VIS.TEXT.COLOR_SOURCE,
#         cfg.VIS.TEXT.COLOR,
#         cfg.VIS.TEXT.PALETTE,
#         instance
#     )
#     for i, line in enumerate(lines):
#         dy = i * (text_h + cfg.VIS.TEXT.SCALE * cfg.VIS.TEXT.LINE_SPACE)
#         cv2.putText(
#             image,
#             line,
#             (x, y+dy),
#             FONT,
#             cfg.VIS.TEXT.SCALE,
#             text_color,
#             cfg.VIS.TEXT.THICKNESS,
#             cv2.LINE_AA
#         )
#     return image


def draw_bounding_box(
    image: np.ndarray,
    bounding_box: BoundingBox,
    color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 1,
    thickness: int = 1,
    fill: bool = False,
    fx: Optional[float] = None,
    fy: Optional[float] = None
) -> np.ndarray:
    """Draw a bounding box in an image.

    Args:
        image (np.ndarray): _description_
        bounding_box (BoundingBox): _description_
        color (Tuple[int, int, int], optional): _description_. Defaults to (255, 255, 255).
        alpha (float, optional): _description_. Defaults to 1.
        thickness (int, optional): _description_. Defaults to 1.
        fill (bool, optional): _description_. Defaults to False.
        fx (Optional[float], optional): _description_. Defaults to None.
        fy (Optional[float], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    thickness = -1 if fill else thickness

    if alpha <= 0:
        return image

    if (fx is not None and fx != 1) or (fy is not None and fy != 1):
        bounding_box = bounding_box.resize(fx=fx, fy=fy)

    xmin, ymin, xmax, ymax = bounding_box.to(
        BoundingBoxFormat.XYXY,
        CoordinatesType.ABSOLUTE,
        image_width=image.shape[1],
        image_height=image.shape[0]
    ).numpy().astype(int)

    if alpha < 1:
        box_mask = cv2.rectangle(
            np.zeros_like(image),
            (xmin, ymin),
            (xmax, ymax),
            (1, 1, 1),
            thickness
        )
        color_mask = np.clip(box_mask * color, 0, 255).astype("uint8")
        image[box_mask == 1] = cv2.addWeighted(
            image[box_mask == 1],
            1-alpha,
            color_mask[box_mask == 1],
            alpha,
            0
        )[:, 0]
    else:
        image = cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color,
            thickness
        )

    return image


def draw_instance(
    image: np.ndarray,
    instance: Instance,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cfg: Optional[DictConfig] = None,
    **kwargs
) -> np.ndarray:
    """_summary_

    Args:
        image (np.ndarray): _description_
        instance (Instance): _description_
        fx (Optional[float], optional): _description_. Defaults to None.
        fy (Optional[float], optional): _description_. Defaults to None.
        cfg (Optional[DictConfig], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    if kwargs:
        cfg = update_copy(cfg, kwargs)

    if instance.bounding_box is not None and cfg.VIS.BOX.VISIBLE:
        box_color = get_color(
            color_source=cfg.VIS.BOX.COLOR_SOURCE,
            color=cfg.VIS.BOX.COLOR,
            palette=cfg.VIS.BOX.PALETTE,
            instance=instance,
        )
        draw_bounding_box(
            image=image,
            bounding_box=instance.bounding_box,
            color=box_color,
            alpha=cfg.VIS.BOX.ALPHA,
            thickness=cfg.VIS.BOX.THICKNESS,
            fill=cfg.VIS.BOX.FILL,
            fx=fx,
            fy=fy
        )
    # if


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> np.ndarray:
    """Resize an image to the given width and height.

    Args:
        image (np.ndarray): The image to resize.
        width (Optional[int], optional): Desired width.
            If None it will resize the image with the provided ``height``,
            maintaining the aspect ration. Defaults to None.
        height (Optional[int], optional): Desired height.
            If None it will resize the image with the provided ``width``,
            maintaining the aspect ration. Defaults to None.

    Returns:
        np.ndarray: The resized image.
    """
    if width is None and height is None:
        return image
    if width is not None and height is not None:
        return cv2.resize(image, (width, height))
    h, w, = image.shape[:2]
    if width is not None:
        f = width / w
    elif height is not None:
        f = height / h
    return cv2.resize(image, None, fx=f, fy=f)


def get_image(dataset_image: Image, cfg: DictConfig) -> np.ndarray:
    """Get a numpy image from an Image object.

    Args:
        dataset_image (Image): The Image object.
        cfg (DictConfig): A  cfg.

    Raises:
        ValueError: If the image can not be read correctly or the size of 
            can not be determined.

    Returns:
        np.ndarray: A numpy image.
    """
    if cfg.VIS.IMG_BACKGROUND:
        image = cv2.imread(str(dataset_image.path))
        if image is None:
            raise ValueError(f"Image is None ({dataset_image.path})")
        return image
    if cfg.VIS.IMG_WIDTH is not None and cfg.VIS.IMG_HEIGHT is not None:
        return np.full(
            (cfg.VIS.IMG_HEIGHT, cfg.VIS.IMG_WIDTH, 3),
            cfg.VIS.IMG_BG_COLOR,
            dtype="uint8"
        )
    if dataset_image.width is not None and dataset_image.height is not None:
        return np.zeros(
            (dataset_image.height, dataset_image.width, 3),
            cfg.VIS.IMG_BG_COLOR,
            dtype="uint8"
        )
    if dataset_image.path.exists():
        image = cv2.imread(str(dataset_image.path))
        if image is None:
            raise ValueError(f"Image is None ({dataset_image.path})")
        return np.full_like(image, cfg.VIS.IMG_BG_COLOR)
    raise ValueError("Can not determine the image size!")


def draw_image_annotations(
    dataset_image: Image,
    cfg: Optional[DictConfig] = None
) -> np.ndarray:
    """Draw the annotations of an Image object.

    Args:
        dataset_image (Image): An Image object.
        cfg (Optional[DictConfig], optional): A cfg. If None, the default
            one will be used. Defaults to None.

    Returns:
        np.ndarray: The image with the annotations drawn.
    """
    image = get_image(dataset_image, cfg)
    
    h, w = image.shape[:2]
    if ((cfg.VIS.IMG_WIDTH is not None and cfg.VIS.IMG_WIDTH != w) or
            (cfg.VIS.IMG_HEIGHT is not None and cfg.VIS.IMG_HEIGHT != h)):
        image, fx, fy = resize_image(
            image, cfg.VIS.IMG_WIDTH, cfg.VIS.IMG_HEIGHT)
    
    annotations = dataset_image.annotations
    for annot in annotations:
        draw_instance(image=image, instance=annot, fx=fx, fy=fy, cfg=cfg)

    return image


# def draw_mask(
#     image: np.ndarray,
#     mask: SegmentationMask,
#     color: Optional[Tuple[int, int, int]] = (0, 0, 255),
#     alpha: float = 0.5,
#     color_by_label: bool = False,
#     label_id: Optional[int] = None,
#     colors_list: Optional[List[Tuple[int, int, int]]] = None
# ) -> np.ndarray:
#     """Draw a segmentation mask on an image.

#     Args:
#         image (np.ndarray): A BGR uint8 image of shape (H, W, 3).
#         mask (SegmentationMask): A SegmentationMask object.
#         color (Optional[Tuple[int, int, int]]): Color of the mask.
#             Defaults to (0,0,255).
#         alpha (float, optional): Transparency of the mask, where
#             1.0 is completely opaque and 0 is transparent. Defaults to 0.5.
#         color_by_label (bool, optional): Set the mask color by its ``label_id``.
#             Defaults to False.
#         label_id (Optional[int], optional): label id associated with the mask;
#             used to chose the color. Defaults to None.
#         colors_list (Optional[List[Tuple[int, int, int]]], optional): Custom
#             list of colors to chose when ``color_by_label`` is set to True.
#             If it is None, a seaborn palette is used. Defaults to None.

#     Returns:
#         np.ndarray: The image with the plotted mask.
#     """
#     mask = mask.mask
#     mask_pixels = image[mask]

#     if color_by_label:
#         if colors_list is None:
#             color = sns.color_palette(None, label_id+1)[label_id]
#             color_mask = np.full_like(mask_pixels, [int(i*255) for i in color])
#         else:
#             color_mask = np.full_like(mask_pixels, colors_list[label_id])
#     else:
#         color_mask = np.full_like(mask_pixels, color)

#     image[mask] = cv2.addWeighted(mask_pixels, 1-alpha, color_mask, alpha, 0)

#     return image


# def draw_coco_keypoints(
#     image: np.ndarray,
#     keypoints: Keypoints.COCOKeypoints,
#     color: Optional[Tuple[int, int, int]] = (0, 0, 255),
#     color_by_label: bool = False,
#     color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
#     keypoint_radius: Optional[int] = 5,
#     connection_rules: Optional[List[Tuple[str,
#                                           str, Tuple[int, int, int]]]] = None,
#     line_thickness: int = 3,
#     show_names: bool = False,
#     show_conf: bool = True,
#     show_keypoints: bool = True,
#     show_connections: bool = True,
#     text_scale: int = 2,
#     text_thickness: int = 2,
#     text_color: Tuple[int, int, int] = (255, 255, 255),
#     text_bg_color: Optional[Tuple[int, int, int]] = None,
#     text_bg_alpha: Optional[float] = None
# ) -> np.ndarray:
#     """Draw keypoints and its connections on an image.

#     Args:
#         image (np.ndarray): A BGR uint8 image of shape (H, W, 3).
#         keypoints (Keypoints.COCOKeypoints): A Keypoints object.
#         color (Optional[Tuple[int, int, int]], optional): Color of the
#             keypoints. Defaults to None.
#         color_by_label (bool, optional): Set the keypoints colors by its label.
#             Defaults to False.
#         color_mapping (Optional[Dict[str, Tuple[int, int, int]]], optional):
#             Mapping between keypoint names and its color. Defaults to None.
#         keypoint_radius (Optional[int], optional): Radius of the keypoints.
#             Defaults to 5.
#         connection_rules (Optional[List[Tuple[str, str, Tuple[int, int, int]]]],
#             optional): List of connections rules
#             (keypoints_name_a, keypoints_name_b, (B, G, R)) Defaults to None.
#         line_thickness (int, optional): Keypoints line thickness. Defaults to 3.
#         show_names (bool, optional): Show the names of the keypoints.
#             Defaults to False.
#         show_conf (bool, optional): Show the keypoint confidence along with its
#             name. Defaults no True.
#         show_keypoints (bool, optional): Show the keypoints circles.
#             Defaults to True.
#         show_connections (bool, optional): Show the connection lines.
#             Defaults to True.
#         text_scale (int, optional): Text scale. Defaults to 2.
#         text_thickness (int, optional): Text thickness. Defaults to 2.
#         text_color (Tuple[int, int, int], optional): Text color.
#             Defaults to (255,255,255).
#         text_bg_color (Optional[Tuple[int, int, int]], optional): Color of the
#             text background. Defaults to None.
#         text_bg_alpha (Optional[float], optional): Transparency of the text
#             background, where 1.0 is completely opaque and 0 is transparent.
#             Defaults to None.

#     Returns:
#         np.ndarray: The image with the plotted mask.
#     """
#     visible_kps = Keypoints.keypoints_dict_to_absolute(
#         keypoints.visible_keypoints,
#         image.shape[1],
#         image.shape[0]
#     )

#     # Draw keypoints connections
#     if show_connections:
#         assert connection_rules is not None
#         for (na, nb, ab_color) in connection_rules:
#             if na in visible_kps and nb in visible_kps:
#                 (xa, ya, ca) = visible_kps[na]
#                 (xb, yb, cb) = visible_kps[nb]
#                 image = cv2.line(
#                     image,
#                     (int(xa), int(ya)),
#                     (int(xb), int(yb)),
#                     ab_color if color_by_label else color,
#                     line_thickness
#                 )
#     # Draw keypoints
#     for name, (x, y, conf) in visible_kps.items():
#         pos = (int(x), int(y))
#         if show_keypoints:
#             image = cv2.circle(
#                 image,
#                 pos,
#                 keypoint_radius,
#                 color_mapping[name] if color_by_label else color,
#                 -1
#             )
#         if show_names:
#             text = f"{name}"
#             if show_conf:
#                 text += f"({conf:.2f})"
#             draw_text(
#                 image=image,
#                 text=text,
#                 position=pos,
#                 color=text_color,
#                 scale=text_scale,
#                 thickness=text_thickness,
#                 background=text_bg_color is not None,
#                 bg_color=text_bg_color,
#                 bg_alpha=text_bg_alpha
#             )
#     return image
