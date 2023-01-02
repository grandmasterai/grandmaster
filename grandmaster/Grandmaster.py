# TODO delete this file

from enum import Enum
from typing import Optional

"""
tasks = Enum(
    "text-classification",
    "image-classification",
    "face-identification",
    "face-features-identification",
    "text-to-image",
    "object-detection",
)
"""


class Grandmaster:
    def __init__(self):
        self.task = None

    def input(self, x):
        pass

    def output(self, x):
        pass

    def get_task(self, input, output, domain: Optional[any] = None):
        if (input.name == "image") and (output.name == "labels"):
            self.task = "image-classification"
            self.input = input
            self.output = output
        elif (input.name == "text") and (output.name == "labels"):
            self.task = "text-classification"
            self.input = input
            self.output = output
        elif (
            input.name == "image"
            and input.representation == "car"
            and output.name == "boundingbox"
            and output.representation == "plate"
        ):
            self.task = "plate-detection"
            self.input = input
            self.output = output
        elif (
            input.name == "image"
            and output.name == "boundingbox"
            and output.representation == "face"
        ):
            self.task = "face-recognition"
            self.input = input
            self.output = output
        elif (
            input.name == "prompt"
            and domain.name == "scientific"
            and output.name == "generated"
        ):
            self.task = "text-generation-scientific"
            self.input = input
            self.output = output
        elif input.name == "audio" and output.name == "text":
            self.task = "speech-recognition"
            self.input = input
            self.output = output

    def inference(self, x, model: Optional[str] = None):
        if self.task == "image-classification":
            from transformers import CLIPProcessor, CLIPModel

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

            inputs = processor(
                text=self.output.values,
                images=x,
                return_tensors="pt",
                padding=True,
            )

            outputs = model(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            probs = logits_per_image.softmax(
                dim=1
            )  # we can take the softmax to get the label probabilities

            print(probs)
        if self.task == "text-classification":
            from transformers import pipeline

            classifier = pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )
            res = classifier(x, self.output.values)
            print(res)
        if self.task == "plate-detection":
            from transformers import YolosFeatureExtractor, YolosForObjectDetection

            feature_extractor = YolosFeatureExtractor.from_pretrained(
                "nickmuchi/yolos-small-rego-plates-detection"
            )
            model = YolosForObjectDetection.from_pretrained(
                "nickmuchi/yolos-small-rego-plates-detection"
            )
            inputs = feature_extractor(images=x, return_tensors="pt")
            outputs = model(**inputs)

            logits = outputs.logits
            bboxes = outputs.pred_boxes

            print(bboxes)
        if self.task == "face-recognition":
            import face_recognition
            import numpy as np

            face_locations = face_recognition.face_locations(np.array(x))
            print(face_locations)

        if self.task == "text-generation-scientific":
            from transformers import pipeline

            model = pipeline("text-generation", model="facebook/galactica-125m")
            print(model(x))

        if self.task == "speech-recognition":
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe("audio.mp3")
            print(result["text"])
