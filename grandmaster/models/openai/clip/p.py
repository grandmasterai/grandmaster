import io
from typing import List, Literal, get_args, Dict, Any
from PIL import Image as PILImage
from pydantic import BaseModel


class Image:
    data: bytes

    def to_pil(self) -> PILImage.Image:
        return PILImage.open(io.BytesIO(self.data))


class CandidateLabel:
    label: str


class InputImageZeroShot:
    image: Image
    candidate_labels: List[CandidateLabel]


class OutputLabel(BaseModel):
    label: str
    score: float


class TASK:
    ZERO_SHOT_IMAGE_CLASSIFICATION = "ZERO_SHOT_IMAGE_CLASSIFICATION"


OutputImageZeroShot = List[OutputLabel]


MODEL_NAME_TYPE = Literal[
    "openai/clip-vit-base-patch16",
    # "openai/clip-vit-base-patch32",
    # "openai/clip-vit-large-patch14",
    # "openai/clip-vit-large-patch14-336",
]

MODEL_NAMES: List[MODEL_NAME_TYPE] = list(get_args(MODEL_NAME_TYPE))


class TaskZeroShot(Task):
    name = TASK.ZERO_SHOT_IMAGE_CLASSIFICATION

    def __init__(self):
        self.engines: Dict[
            MODEL_NAME_TYPE, Dict[Literal["model", "processor"], Any]
        ] = {}

    def setup(self):
        from transformers import CLIPProcessor, CLIPModel

        for n in MODEL_NAMES:
            self.engines[n] = {
                "model": CLIPModel.from_pretrained(n),
                "processor": CLIPProcessor.from_pretrained(n),
            }

    def predict(
        self,
        query: InputImageZeroShot,
        model_name: MODEL_NAME_TYPE = "openai/clip-vit-base-patch16",
        model_settings={},
    ) -> OutputImageZeroShot:

        image: PILImage.Image = query.image.to_pil()

        model = self.engines[model_name]["model"]
        processor = self.engines[model_name]["processor"]

        processed = processor(
            text=query.candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        x = model(**processed)
        results = list(x.logits_per_image.softmax(dim=1).detach().numpy()[0])

        out: List[OutputLabel] = []
        for i, r in zip(query.candidate_labels, results):
            out.append(OutputLabel(label=i, score=r.item()))
        return out
