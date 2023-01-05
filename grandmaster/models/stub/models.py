from grandmaster.Model import Model

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
            "dataType": "LABEL",
        }

        self.model_name = "nickmuchi/yolos-small-rego-plates-detection"

    def tasks(self):
        return [create_task(self.inputs, self.outputs, self.model, self.postprocess)]

    def model(self, query: ImageQueryTypedDict) -> List[ResultBoundingBoxBoxDC]:  # TODO
        image = load_image_from_data(query["image"])
        self.feature_extractor(images=image, return_tensors="pt")
        out = self.modelx(**x)  # type: ignore
        # outputs.pred_boxes
        return []

    def postprocess(self, x: Any) -> List[ResultBoundingBoxBoxDC]:
        return x
