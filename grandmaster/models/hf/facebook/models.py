import torch
from typing import Any, List
from grandmaster.Model import Model
from grandmaster.Task import Task, create_task
from grandmaster.helper import load_image_from_data

from grandmaster.proto.mytypes import (
    InputsTypedDict,
    OutputsTypedDict,
    PromptQueryTypedDict,
    ImageQueryTypedDict,
    InputsType,
    ResultBoundingBoxBoxDC,
    ResultBoundingBoxDC,
    ResultGeneratedTextDC,
    ResultLabelDC,
    TextZeroShotQueryTypedDict,
)


class BART(Model):
    # ZeroShotTextClassification
    def __init__(self):
        self.model_name = "facebook/bart-large-mnli"
        self.inputs: InputsTypedDict = {"dataType": "TEXT"}
        self.outputs: OutputsTypedDict = {"dataType": "LABEL"}

    def tasks(self):
        return [create_task(self.inputs, self.outputs, self.model, self.postprocess)]

    def model(self, query: TextZeroShotQueryTypedDict) -> Any:
        from transformers import pipeline

        classifier = pipeline(
            task="zero-shot-classification", model="facebook/bart-large-mnli"
        )

        results = classifier(query["text"], query["candidate_labels"].split(","))
        labels: List[str] = results["labels"]  # type: ignore
        scores: List[float] = results["scores"]  # type: ignore

        out: List[ResultLabelDC] = []
        for (l, s) in zip(labels, scores):
            out.append(ResultLabelDC(label=l, score=s))
        return out

    def postprocess(self, x: Any) -> List[ResultLabelDC]:
        return x


class Galactica(Model):
    # LM Scientific
    # TODO add 'SCIENTIFIC' domain information
    def __init__(self):
        self.inputs: InputsTypedDict = {"dataType": "PROMPT"}
        self.outputs: OutputsTypedDict = {"dataType": "COMPLETION"}
        self.model_name = "facebook/galactica-125m"

    def tasks(self):
        return [create_task(self.inputs, self.outputs, self.model, self.postprocess)]

    def model(self, query: PromptQueryTypedDict) -> ResultGeneratedTextDC:
        from transformers import pipeline

        model = pipeline(task="text-generation", model="facebook/galactica-125m")
        text = query["text"]
        generated: str = model(text)  # type: ignore
        return ResultGeneratedTextDC(text=generated)

    def postprocess(self, x: Any) -> ResultGeneratedTextDC:
        return x


class Detr(Model):
    # ZeroShotObject Detection
    def __init__(self):
        self.model_name = "facebook/detr-resnet-50"

        from transformers import DetrFeatureExtractor, DetrForObjectDetection

        self.feature_extractor = lambda: DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        self.model = lambda: DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        )

        self.loaded_feature_extractor = None
        self.loaded_model = None

        self.inputs: InputsTypedDict = {"dataType": "IMAGE"}
        self.outputs: OutputsTypedDict = {"dataType": "BOUNDINGBOX"}

    def tasks(self):
        return [create_task(self.inputs, self.outputs, self.apply, self.postprocess)]

    def apply(self, query: ImageQueryTypedDict) -> List[ResultBoundingBoxDC]:

        model = self.get_model()
        feature_extractor = self.get_feature_extractor()

        image = load_image_from_data(query["image"])

        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)  # type: ignore
        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process_object_detection(
            outputs, target_sizes=target_sizes
        )[0]

        out: List[ResultBoundingBoxDC] = []

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            if score > 0.9:  # TODO
                label_name = model.config.id2label[label.item()]  # type: ignore
                box = [round(i, 2) for i in box.tolist()]
                box = ResultBoundingBoxBoxDC(
                    xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3]
                )
                score = round(score.item(), 4)
                out.append(ResultBoundingBoxDC(score=score, label=label_name, box=box))

        return out

    def get_model(self):
        if self.loaded_model:
            return self.loaded_model
        else:
            self.loaded_model = self.model()
            return self.loaded_model

    def get_feature_extractor(self):
        if self.loaded_feature_extractor:
            return self.loaded_feature_extractor
        else:
            self.loaded_feature_extractor = self.feature_extractor()
            return self.loaded_feature_extractor

    def postprocess(self, x: Any) -> List[ResultBoundingBoxDC]:
        return x
