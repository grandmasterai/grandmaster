from grandmaster.inputs import Image, Audio, Label
from typing import List, Protocol


class PredictProtocol(Protocol):
    def predict(self, inputs: Image) -> List[Label]:
        ...


class ImageClassificationTask(PredictProtocol):
    task_name = "image-classification"

    def predict(self, image: Image) -> List[Label]:
        ...


class AudioClassificationTask(PredictProtocol):
    task_name = "audio-classification"

    def predict(self, audio: Audio) -> List[Label]:
        ...


class Model1:
    task = ImageClassificationTask

    def predict(self, image: Image) -> List[Label]:
        return [Label(label="cat")]


class Model2:
    task = ImageClassificationTask

    def predict(self, audio: Audio) -> List[Label]:
        return [Image(image="cat")]


all_models = [Model1, Model2]

for model in all_models:
    if not isinstance(model(), model.task):
        print(model)
