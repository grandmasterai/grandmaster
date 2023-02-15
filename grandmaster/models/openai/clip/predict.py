from typing import List
from grandmaster.Task import Task
from grandmaster.inputs import Image, Label
from grandmaster.outputs import LabelWithScore
from grandmaster.tasks import ZeroShotImageClassification

from transformers import pipeline

"""
class CLIP(Model):
    # image classification
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel

        self.xmodel = lambda: CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.xprocessor = lambda: CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )

        self.model_name = "openai/clip-vit-base-patch16"

        self.inputs: List[InputsTypedDict] = [
            {"dataType": "IMAGE"},
            {"dataType": "TEXT"},
        ]
        self.outputs: List[OutputsTypedDict] = [{"dataType": "LABEL"}]

    def load(self):
        self.xmodel()
        self.xprocessor()

    def tasks(self):
        return [
            create_task(
                self.model_name,
                self.inputs,
                self.outputs,
                self.apply,
                self.preprocess,
                self.postprocess,
            )
        ]

    def apply(self, query: ImageZeroShotQueryTypedDict) -> Any:
        image = load_image_from_data(query["image"])
        processed = self.xprocessor()(
            text=query["candidate_labels"].split(","),
            images=image,
            return_tensors="pt",
            padding=True,
        )
        x = self.xmodel()(**processed)  # type: ignore
        results = list(x.logits_per_image.softmax(dim=1).detach().numpy()[0])
        out: List[ResultLabelDC] = []
        for i, r in zip(query["candidate_labels"].split(","), results):
            out.append(ResultLabelDC(label=i, score=r.item()))
        return out

    def postprocess(self, x: Any) -> List[ResultLabelDC]:
        return x

    def preprocess(self, query: ImageZeroShotQueryTypedDict) -> Any:
        return query
"""

class CLIP:
    task = ZeroShotImageClassification
    model_names = ["openai/clip-vit-large-patch14-336"]

    def load_model(self):
        model_name = "openai/clip-vit-large-patch14-336"
        self.classifier = pipeline("zero-shot-image-classification", model=model_name)

    def predict(
        self, image: Image, candidate_labels: List[Label]
    ) -> List[LabelWithScore]:
        out = self.classifier(image, candidate_labels=candidate_labels)
        return []

class CLIPForImageEmbeddings(Task):
