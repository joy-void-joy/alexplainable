# MIT License
# Adapted from https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py

# %%
import torch
import pydantic

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image


class LLaVA:
    class LLaVAConfig(pydantic.BaseModel):
        llava_id: str = "liuhaotian/llava-v1.6-vicuna-7b"
        llava_base: str | None = None
        load_8bit: bool = False
        load_4bit: bool = False
        device: str = "cuda"
        max_new_tokens: int = 100
        temperature: float = 0

    config: LLaVAConfig
    conv_mode: str

    model_name: str

    def __init__(
        self,
        config: LLaVAConfig = LLaVAConfig(),
    ):
        self.config = config
        self.model_name = get_model_name_from_path(self.config.llava_id)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.config.llava_id,
            self.config.llava_base,
            self.model_name,
            load_8bit=self.config.load_8bit,
            load_4bit=self.config.load_4bit,
            device=self.config.device,
        )
        self.conv_mode = self.infer_conv_mode(self.model_name.lower())

    def infer_conv_mode(self, name):
        if "llama-2" in name:
            return "llava_llama_2"
        elif "mistral" in name:
            return "mistral_instruct"
        elif "v1.6-34b" in name:
            return "chatml_direct"
        elif "v1" in name:
            return "llava_v1"
        elif "mpt" in name:
            return "mpt"
        else:
            return "llava_v0"

    def infer(self, image: Image.Image, prompt: str):
        conv = conv_templates[self.conv_mode].copy()

        image = image.convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)

        if isinstance(image_tensor, list):
            image_tensor = [
                image.to(self.model.device, dtype=torch.float16)
                for image in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        if self.model.config.mm_use_im_start_end:
            inp = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt
            )
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        input_ids = (
            tokenizer_image_token(
                conv.get_prompt(),
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True if self.config.temperature > 0 else False,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,
            )

            return (
                self.tokenizer.decode(output_ids[0])
                .removeprefix("<s>")
                .removesuffix("</s>")
                .strip()
            )


# %%
