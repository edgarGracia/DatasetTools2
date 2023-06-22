from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import seaborn as sns
from omegaconf import DictConfig
import seaborn as sns

from DatasetTools.config.config import get_cfg, update_copy
from DatasetTools.utils.utils import ColorSource, RelativePosition
from DatasetTools.structures.instance import Instance
from DatasetTools.structures.image import Image
from DatasetTools.structures import bounding_box

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


def get_text_position_from_box(
    box: bounding_box.BaseBoundingBox,
    relative_position: RelativePosition
) -> Tuple[int, int]:
    """Get the correct initial position of a text given a bounding box.

    Args:
        box (bounding_box.BaseBoundingBox): A bounding box object.
        relative_position (RelativePosition): In which corner of the box put
            the text.

    Returns:
        Tuple[int, int]: Final text position (x, y).
    """
    xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
    if relative_position is RelativePosition.TOP_LEFT:
        return (xmin, ymin)
    elif relative_position is RelativePosition.TOP_RIGHT:
        return (xmax, ymin)
    elif relative_position is RelativePosition.BOTTOM_LEFT:
        return (xmin, ymax)
    elif relative_position is RelativePosition.BOTTOM_RIGHT:
        return (xmax, ymax)
    else:
        raise NotImplementedError(str(relative_position))


def draw_text(
    image: np.ndarray,
    text: Union[str, List[str]],
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
    thickness: int = 1,
    line_space: int = 15,
    relative_position: RelativePosition = RelativePosition.TOP_RIGHT,
    background: bool = True,
    background_color: Tuple[int, int, int] = (50, 50, 50),
    background_alpha: float = 1,
    margin: int = 0
) -> np.ndarray:
    """Draw text in the given image.

    Args:
        image (np.ndarray): The source image where draw the text.
        text (Union[str, List[str]]): The text to draw. A list of lines or a
            single string. The line breaks ('\n') will be parsed to lines.
        position (Tuple[int, int]): The text position within the image (x, y).
        color (Tuple[int, int, int], optional): The color of the text.
            Defaults to (255, 255, 255).
        scale (int, optional): The scale of the text. Defaults to 1.
        thickness (int, optional): The text thickness_. Defaults to 1.
        line_space (int, optional): Line spacing. Defaults to 15.
        relative_position (RelativePosition, optional): Relative position of
            the text to the provided position. Defaults to
            RelativePosition.TOP_RIGHT.
        background (bool, optional): Whether draw a background area behind the
            actual text. Defaults to True.
        background_color (Tuple[int, int, int], optional): Color of the
            background. Defaults to (50, 50, 50).
        background_alpha (float, optional): Opacity of the background.
            Defaults to 1.
        margin (int, optional): Text margin. Defaults to 0.

    Returns:
        np.ndarray: The source image with the text drawn.
    """
    if not text:
        return image

    text_lines = text.splitlines() if isinstance(text, str) else text
    
    # Compute the final text size
    max_text_len = max(text_lines, key=lambda x: len(x))
    (text_w, text_h), _ = cv2.getTextSize(
        max_text_len,
        CV2_FONT,
        scale,
        thickness
    )
    box_h = text_h + ((text_h + scale * line_space) *
                      (len(text_lines) - 1)) + (margin * 2)

    # Compute the text starting position
    x, y = position
    if relative_position is RelativePosition.TOP_RIGHT:
        pass
    elif relative_position is RelativePosition.TOP_LEFT:
        x -= text_w + (margin*2)
    elif relative_position is RelativePosition.BOTTOM_RIGHT:
        y += box_h
    elif relative_position is RelativePosition.BOTTOM_LEFT:
        x -= text_w + (margin*2)
        y += box_h
    else:
        raise NotImplementedError(str(relative_position))

    # Clip values
    x = min(max(x, 0), image.shape[1] - 1)
    y = min(max(y, 0), image.shape[0] - 1)

    # Draw background
    if background and background_alpha > 0:
        xmin = max(x, 0)
        ymin = max(y - box_h, 0)
        xmax = min(x + text_w + (margin * 2), image.shape[1])
        ymax = min(y, image.shape[0])
        print("\n", ymin, ymax, xmin, xmax, background_alpha, y, image.shape, box_h)
        
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

    # Draw the text lines
    for i, line in enumerate(reversed(text_lines)):
        dy = i * (text_h + scale * line_space)
        cv2.putText(
            image, line, (x + margin, y - margin - dy), CV2_FONT, scale, color,
            thickness, cv2.LINE_AA)

    return image


def draw_bounding_box(
    image: np.ndarray,
    box: bounding_box.BaseBoundingBox,
    color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 1,
    thickness: int = 1,
    fill: bool = False
) -> np.ndarray:
    """Draw a bounding box in an image.

    Args:
        image (np.ndarray): The source image where draw the text.
        box (bounding_box.BaseBoundingBox): A bounding box object.
        color (Tuple[int, int, int], optional): Color of the box.
            Defaults to (255, 255, 255).
        alpha (float, optional): Opacity of the box. Defaults to 1.
        thickness (int, optional): Thickness of the box. Defaults to 1.
        fill (bool, optional): Whether to fill the box. Defaults to False.

    Returns:
        np.ndarray: The source image with the box drawn.
    """
    thickness = -1 if fill else thickness

    if alpha <= 0:
        return image

    xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax

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


def instance_text_formatter(
    instance: Instance,
    expression: str
) -> str:
    """Create the text to display from an instance.

    WARNING: This function uses the ``eval`` function and can execute arbitrary
    code.

    Args:
        instance (Instance): An Instance object.
        expression (str): Expression to evaluate that should return string to show.

    Returns:
        str: The text with the instance values to show.
    """
    return eval(f"str({expression})")

def draw_instance(
    image: np.ndarray,
    instance: Instance,
    scale: Optional[Tuple[float, float]] = None,
    cfg: Optional[DictConfig] = None,
    **kwargs
) -> np.ndarray:
    """Draw an instance object over a given image.

    Args:
        image (np.ndarray): Numpy image of shape (H, W, 3) and "uint8" dtype.
        instance (Instance): The Instance object to draw.
        scale (Optional[Tuple[float, float]]): The scale factor (x, y), used to
            scale the instance's structures in case the image was resized.
            Defaults to None.
        cfg (Optional[DictConfig], optional): A configuration object.
            Defaults to None.
        **kwargs: Arguments to override the cfg.

    Returns:
        np.ndarray: The source image with the instance's data drawn.
    """
    if kwargs:
        cfg = update_copy(cfg, kwargs)

    box = instance.bounding_box

    # Draw bounding box
    if box is not None:
        if scale is not None:
            box = box.scale(*scale)
        
        if cfg.VIS.BOX.VISIBLE:
            box_color = get_color(
                color_source=cfg.VIS.BOX.COLOR_SOURCE,
                color=cfg.VIS.BOX.COLOR,
                palette=cfg.VIS.BOX.PALETTE,
                instance=instance,
            )
            draw_bounding_box(
                image=image,
                box=box,
                color=box_color,
                alpha=cfg.VIS.BOX.ALPHA,
                thickness=cfg.VIS.BOX.THICKNESS,
                fill=cfg.VIS.BOX.FILL
            )

        # Draw text along the box
        if cfg.VIS.TEXT.VISIBLE and cfg.VIS.TEXT.FORMATTER:
            text = instance_text_formatter(instance, cfg.VIS.TEXT.FORMATTER)
            position = get_text_position_from_box(
                box=box,
                relative_position=cfg.VIS.BOX.TEXT_POSITION
            )
            print(position)
            text_color = get_color(
                color_source=cfg.VIS.TEXT.COLOR_SOURCE,
                color=cfg.VIS.TEXT.COLOR,
                palette=cfg.VIS.TEXT.PALETTE,
                instance=instance,
            )
            text_bg_color = get_color(
                color_source=cfg.VIS.TEXT_BG.COLOR_SOURCE,
                color=cfg.VIS.TEXT_BG.COLOR,
                palette=cfg.VIS.TEXT_BG.PALETTE,
                instance=instance,
            )
            draw_text(
                image=image,
                text=text,
                position=position,
                color=text_color,
                scale=cfg.VIS.TEXT.SCALE,
                thickness=cfg.VIS.TEXT.THICKNESS,
                line_space=cfg.VIS.TEXT.LINE_SPACE,
                relative_position=cfg.VIS.TEXT.POSITION,
                background=cfg.VIS.TEXT_BG.VISIBLE,
                background_color=text_bg_color,
                background_alpha=cfg.VIS.TEXT_BG.ALPHA,
                margin=cfg.VIS.TEXT_BG.MARGIN
            )


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Tuple[np.ndarray, float, float]:
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
        Tuple(np.ndarray, float, float): The resized image, the horizontal
            resize factor and the vertical resize factor.
    """
    if width is None and height is None:
        return image, 1, 1
    h, w, = image.shape[:2]
    if width is not None and height is not None:
        return cv2.resize(image, (width, height)), width / w, height / h
    if width is not None:
        f = width / w
    elif height is not None:
        f = height / h
    return (cv2.resize(image, None, fx=f, fy=f), f, f)


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
        cfg (Optional[DictConfig], optional): A configuration object.
            If None, the default will be used. Defaults to None.

    Returns:
        np.ndarray: An image with the annotations drawn.
    """
    image = get_image(dataset_image, cfg)
    image_scale = None
    
    h, w = image.shape[:2]
    if ((cfg.VIS.IMG_WIDTH is not None and cfg.VIS.IMG_WIDTH != w) or
            (cfg.VIS.IMG_HEIGHT is not None and cfg.VIS.IMG_HEIGHT != h)):
        image, fx, fy = resize_image(
            image, cfg.VIS.IMG_WIDTH, cfg.VIS.IMG_HEIGHT)
        image_scale = (fx, fy)
    
    annotations = dataset_image.annotations
    for annot in annotations:
        draw_instance(image=image, instance=annot, scale=image_scale, cfg=cfg)

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
