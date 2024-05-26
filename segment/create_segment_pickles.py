from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import PIL.Image
import pydantic
import numpy as np
import cv2
import pathlib
from tqdm import tqdm
import pickle
from natsort import natsorted

data_dir = pathlib.Path("../data/imagenet/ILSVRC/Data/CLS-LOC/train/n11939491")
output_dir = pathlib.Path("../data/segments")
output_dir.mkdir(parents=True, exist_ok=True)


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


class SAM:
    checkpoint = "../data/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    model = SamAutomaticMaskGenerator(
        sam_model_registry[model_type](checkpoint=checkpoint).to(device="cuda"),
        points_per_side=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )


sam = SAM()


def get_masks(image: PIL.Image.Image):
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return Masks(masks=sam.model.generate(cv2_image))


for image_path in data_dir.glob("*.JPEG"):
    with open(
        output_dir / f"{image_path.with_suffix('.pickle').name}",
        "wb",
    ) as handle:
        pickle.dump(get_masks(PIL.Image.open(image_path)), handle)
