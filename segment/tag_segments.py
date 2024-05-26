# %%
from tqdm import tqdm
import pathlib
import PIL.Image
from .llava_encapsulate import LLaVA
import random


def shuffled(iterable):
    return random.sample(iterable, len(iterable))


llava = LLaVA(config=LLaVA.LLaVAConfig(llava_id="liuhaotian/llava-v1.5-7b"))

pickle_path = pathlib.Path("../data/n11939491/segments")

categories = [
    "petal",
    "flower head",
    "whole flower",
    "center disk",
    "stem",
    "leaf",
    "other",
]
prompt = f"""Please categorize the part highlighted in red as one of: {','.join(categories)}. If the segment is not part of a daisy, answer none.
Only answer with the category."""


def validate_size(image: PIL.Image.Image, size: int = 25):
    return image.size[0] >= size and image.size[1] >= size


for mask_file in tqdm(
    shuffled(
        [
            mask
            for mask_dir in pickle_path.glob("*")
            if mask_dir.is_dir()
            for mask in mask_dir.iterdir()
            if mask.is_file()
        ]
    )
):
    mask_image = PIL.Image.open(mask_file).convert("RGB")
    if not validate_size(mask_image := PIL.Image.open(mask_file)):
        continue
    category = llava.infer(mask_image, prompt).lower()
    if category in categories:
        outdir = mask_file.parent / "sorted" / category
        outdir.mkdir(parents=True, exist_ok=True)
        if not (outdir / mask_file.name).exists():
            (outdir / mask_file.name).symlink_to(mask_file)
