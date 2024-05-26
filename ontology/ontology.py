# %%
from typing import Any
from tqdm import tqdm

import pickle
from natsort import natsorted
import pydantic
import io
import base64
import PIL.Image
import anthropic
import pathlib

client = anthropic.Anthropic(
    api_key=pathlib.Path("../.anthropic_key").read_text().strip()
)

data_dir = pathlib.Path("../data/imagenet/ILSVRC/Data/CLS-LOC/train/n03126707")


class ClaudePrompt(pydantic.BaseModel):
    messages: Any
    system: str | None = None

    class Config:
        arbitrary_types_allowed = True


initial_ontology = """
{
  sky: {
    dirty: {},
    'clear sky': {
      blue: {},
    },
    clouds: {},
    sunset: {
      orange: {},
    },
  },
  crane: {
    'yellow/orange crane': {
      yellow: {},
      orange: {},
     },
     'black crane': {
      black: {},
    },
    wireframe: {},
    glass: {},
    logo: {},
    cables: {},
    'tracks/wheels': {
      tracks: {
        iron: {
          rusty: {},
        },
      },
      wheels: {
        'two wheels': {
          wheel: {
            'tire': {},
            'hubcap': {}
          },
        },
        'six wheels': {
          'wheel': {},
        },
      },
    },
    'rearview mirror': {},
    tank: {},
  },
}
"""
current_ontology = initial_ontology


def claude_prompt(prompt: ClaudePrompt, path: pathlib.Path, recompute=False):
    if path.exists() and not recompute:
        return path.read_text()

    path.with_suffix(".prompt").write_text(prompt.model_dump_json(indent=2))

    call = client.messages.create(
        **dict(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0,
            messages=prompt.messages,
        )
        | ({"system": prompt.system} if prompt.system else {})
    )
    result = call.content[0].text

    with open(path.with_suffix(".pickle"), "wb") as f:
        pickle.dump(call, f)
    path.write_text(result)

    return result


def get_64(im: PIL.Image):
    buffered = io.BytesIO()
    im.save(buffered, format="PNG")

    return {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(buffered.getvalue()).decode("utf-8"),
    }


def prompt_description(image: PIL.Image) -> ClaudePrompt:
    return ClaudePrompt(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe everything you see in this image, be meticulous.\n",
                    }
                ]
                + [{"type": "image", "source": get_64(image)}],
            }
        ]
    )


def prompt_extract_json(description: str) -> ClaudePrompt:
    return ClaudePrompt(
        system=f"""Here is a JSON for an image recognizer:

{initial_ontology}

Please propose a JSON of features appearing in the image of the description under the same format (All values must be objects). Ignore any comments that are speculative and are not part of the image.
Be sure to have a top-level node for crane. Answer only in JSON""",
        messages=[{"role": "user", "content": [{"type": "text", "text": description}]}],
    )


def prompt_merge_json(json1: str, json2: str) -> ClaudePrompt:
    return ClaudePrompt(
        system="""Please refactor the following JSON for image recognition. 
It is a bottom-up construction of features that add up to recognize images.

- Remove any node that are speculative, or are interpretations not directly seen in the image (for instance remove "allows pivoting" or "impressive")
- Each node must be visible on its own and answer the question "can this be seen in this image"
- Each node is composed of other visible features that adds up to compose it
- Remove or generalize too specific nodes

For instance, instead of

"warning lights": {
  "on outriggers": {}
}

you want

"warning lights on outriggers": {
  "warning lights": {},
  "outriggers": {}
}
Because "warning lights" is a visible feature of an image, so is "warning lights on outriggers", and to see "warning lights on outriggers", one need to first detect "warning lights" and "outriggers".

Answer only in JSON""",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{json1}\n\n{json2}"}],
            }
        ],
    )


def prompt_clean_json(description: str) -> ClaudePrompt:
    return ClaudePrompt(
        system="""Please refactor the following JSON for image recognition
It is a bottom-up construction of features that add up to recognize images.

- Each node must be visible on its own and answer the question "can this be seen in this image"
- Each node is composed of other visible features that adds up to compose it

Please put similar nodes into the same parent node, while keeping parent nodes as directly observable features (e.g. not "color" but "red/blue/yellow")

Answer only in JSON.
""",
        messages=[{"role": "user", "content": [{"type": "text", "text": description}]}],
    )


def prompt_modify_graph(image: PIL.Image, ontology: str) -> ClaudePrompt:
    return ClaudePrompt(
        system=f"""Here is a JSON ontology for image recognition:

{ontology}
        
Answer only in JSON""",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The following ontology could not recognize this image:",
                    },
                    {"type": "image", "source": get_64(image)},
                    {
                        "type": "text",
                        "text": "as a crane. Please modify it to incorporate all features of this image into it.",
                    },
                ],
            }
        ],
    )


def prompt_clean_graph(ontology: str) -> ClaudePrompt:
    return ClaudePrompt(
        system="""Please refactor the following JSON for image recognition. 
It is a bottom-up construction of features that add up to recognize images.

- Remove any node that are speculative, or are interpretations not directly seen in the image (for instance remove "allows pivoting" or "impressive")
- Each node must be visible on its own and answer the question "can this be seen in this image"
- Each node is composed of other visible features that adds up to compose it

For instance, instead of

"warning lights": {
  "on outriggers": {}
}

you want

"warning lights on outriggers": {
  "warning lights": {},
  "outriggers": {}
}
Because "warning lights" is a visible feature of an image, so is "warning lights on outriggers", and to see "warning lights on outriggers", one need to first detect "warning lights" and "outriggers".

Answer only in JSON""",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": ontology}],
            }
        ],
    )


for i in tqdm(list(natsorted(data_dir.glob("*.JPEG")))[:10]):
    outdir = pathlib.Path("../../ontology") / i.with_suffix("").name
    outdir.mkdir(exist_ok=True, parents=True)

    description = claude_prompt(
        prompt_description(PIL.Image.open(i)), (outdir / "description")
    )

    extracted_json = claude_prompt(
        prompt_extract_json(description), outdir / "extracted"
    )

    current_ontology = claude_prompt(
        prompt_merge_json(current_ontology, extracted_json),
        outdir / "merged",
        recompute=True,
    )

    print(current_ontology)
