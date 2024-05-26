import PIL.Image
import pydantic
import numpy as np
import cv2
import pathlib
from tqdm import tqdm
import pickle
from natsort import natsorted

pickle_dir = pathlib.Path("../data/n11939491/segments")
n119_dir = pathlib.Path("../data/imagenet/ILSVRC/Data/CLS-LOC/train/n11939491")


class Masks(pydantic.BaseModel):
    class Mask(pydantic.BaseModel):
        segmentation: np.ndarray
        area: float
        bbox: list[int]
        predicted_iou: float
        point_coords: list[list[float]]
        crop_box: list[int]

        class Config:
            arbitrary_types_allowed = True

    masks: list[Mask]


class MaskedImage(pydantic.BaseModel):
    segment: PIL.Image.Image
    cropped: PIL.Image.Image
    highlight: PIL.Image.Image

    class Config:
        arbitrary_types_allowed = True


def get_masked_image(
    image: PIL.Image.Image, mask: Masks.Mask, color=(255, 0, 0), alpha=0.5
):
    image = image.convert("RGBA")
    result_segment = PIL.Image.new(
        "RGBA",
        (mask.bbox[2], mask.bbox[3]),
        (0, 0, 0, 0),
    )
    result_cropped = PIL.Image.new(
        "RGBA",
        (mask.bbox[2], mask.bbox[3]),
        (0, 0, 0, 0),
    )

    bbox = [
        mask.bbox[0],
        mask.bbox[1],
        min(mask.bbox[0] + mask.bbox[2], image.width),
        min(mask.bbox[1] + mask.bbox[3], image.height),
    ]
    image_cropped = image.crop(bbox)

    mask_image = PIL.Image.fromarray(np.uint8(mask.segmentation) * 255)
    colored_mask = PIL.Image.new("RGBA", mask_image.size, color)

    result_segment.paste(image_cropped, mask=mask_image.crop(bbox))
    result_cropped.paste(image_cropped)
    return MaskedImage(
        segment=result_segment,
        cropped=result_cropped,
        highlight=PIL.Image.composite(
            PIL.Image.blend(image, colored_mask, alpha=alpha), image, mask_image
        ),
    )


def save_image(image: PIL.Image.Image, path: pathlib.Path):
    if path.exists():
        return
    image.save(path)


def pickle_to_masks(pickle_path: pathlib.Path):
    with open(pickle_path, "rb") as f:
        masks = pickle.load(f)

    image = PIL.Image.open(n119_dir / pickle_path.with_suffix(".JPEG").name)

    for number, mask in tqdm(list(enumerate(masks.masks)), leave=False):
        output_dir = pickle_path.with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        masked = get_masked_image(image, mask)
        try:
            save_image(masked.segment, output_dir / f"segment_{number}.png")
            save_image(masked.cropped, output_dir / f"cropped_{number}.png")
            # save_image(masked.highlight, output_dir / f"highlight_{number}.png")
        except SystemError:
            pass


for i in tqdm(natsorted(pickle_dir.glob("*.pickle"))):
    pickle_to_masks(i)
