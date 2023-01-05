from typing import Any, List
from grandmaster.Model import Model
from grandmaster.Task import create_task
from grandmaster.helper import load_image_from_data
from grandmaster.proto.mytypes import (
    ImageQueryTypedDict,
    InputsTypedDict,
    OutputsTypedDict,
    ResultBoundingBoxBoxDC,
    ResultBoundingBoxDC,
)


class YOLOPlate(Model):
    def __init__(self):
        from transformers import YolosFeatureExtractor, YolosForObjectDetection

        self.feature_extractor = YolosFeatureExtractor.from_pretrained(
            "nickmuchi/yolos-small-rego-plates-detection"
        )
        self.modelx = YolosForObjectDetection.from_pretrained(
            "nickmuchi/yolos-small-rego-plates-detection"
        )

        self.inputs: InputsTypedDict = {"dataType": "IMAGE"}
        self.outputs: OutputsTypedDict = {
            "dataType": "BOUNDINGBOX",
            "representation": "PLATE",
        }

        self.model_name = "nickmuchi/yolos-small-rego-plates-detection"

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

    def apply(self, query: ImageQueryTypedDict) -> List[ResultBoundingBoxBoxDC]:  # TODO
        image = load_image_from_data(query["image"])
        self.feature_extractor(images=image, return_tensors="pt")
        out = self.modelx(**x)  # type: ignore
        # outputs.pred_boxes
        return []

    def postprocess(self, x: Any) -> List[ResultBoundingBoxBoxDC]:
        return x

    def preprocess(self, x):
        return x


class FaceRecognition(Model):
    def __init__(self):

        self.inputs: InputsTypedDict = {"dataType": "IMAGE"}
        self.outputs: OutputsTypedDict = {
            "dataType": "BOUNDINGBOX",
            "representation": "FACE",
        }
        self.model_name = "community/face_recognition"

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

    def apply(self, query: ImageQueryTypedDict) -> List[ResultBoundingBoxDC]:
        import face_recognition
        import numpy as np

        image = load_image_from_data(query["image"])
        boxes = face_recognition.face_locations(np.array(image))

        out: List[ResultBoundingBoxDC] = []
        for (t, r, b, l) in boxes:
            box = ResultBoundingBoxBoxDC(xmin=l, ymin=t, xmax=r, ymax=b)
            out.append(ResultBoundingBoxDC(score=None, label="face", box=box))
        return out

    def postprocess(self, x: Any) -> List[ResultBoundingBoxBoxDC]:
        return x

    def preprocess(self, x):
        return x
