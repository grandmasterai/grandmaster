from typing import List, Generic, TypeVar, Dict, Union, Type, TypedDict, Literal

from grandmaster.inputs import File, Image, Label
from grandmaster.outputs import LabelWithScore, BoundingBox
from grandmaster.examples import Puppy
from grandmaster.annotations import BaseModel
from grandmaster.Model import Model


from pydantic import Field, create_model
from pydantic.main import ModelMetaclass

from abc import ABC


class TaskTwo(ABC):
    @classmethod
    def get_subclass(cls):
        return cls.__subclasses__()

    name: str

    class Inputs:
        pass

    class Outputs:
        pass

    def predict(
        self,
        input,
    ) -> Outputs:
        ...

    @classmethod
    def create_inputs_with_models(cls):

        models = cls.get_subclass()
        model_names_str: List[str] = [m.model_name for m in models] + ["x"]
        model_names = Literal[tuple(model_names_str)]

        return create_model(
            "InputsWithModels",
            __base__=cls.Inputs,
            model_name=(model_names, model_names_str[0]),
            task_name=(Literal[cls.name], cls.name),
        )


class ImageClassification(TaskTwo):
    name = "image-classification"

    class Inputs(BaseModel):
        image: File = Field(example="http://localhost:3000/ex/puppy.jpeg")
        candidate_labels: List[str] = Field(example=["dog", "cat"])

    class Outputs(BaseModel):
        labels: List[LabelWithScore]

    examples = [{"image": Puppy, "labels": ["dog", "cat"]}]

    default_model_name = "openai/clip-vit-large-patch14-336"


class TextClassification(TaskTwo):
    name = "text-classification"

    class Inputs(BaseModel):
        text: str = Field()

    class Outputs(BaseModel):
        labels: List[LabelWithScore]

    examples = []

    default_model_name = "X"


class FaceRecognition(TaskTwo):
    name = "face-recognition"

    class Inputs(BaseModel):
        image: File = Field(example="http://localhost:3000/ex/obama.jpeg")

    class Outputs(BaseModel):
        results: List[BoundingBox]

    default_model_name = "A"


class ObjectDetection(TaskTwo):
    name = "object-detection"

    class Inputs(BaseModel):
        image: File = Field(example="http://localhost:3000/ex/obama.jpeg")

    class Outputs(BaseModel):
        results: List[BoundingBox]

    default_model_name = "A"


all_tasks: List[Type[TaskTwo]] = [
    ImageClassification,
    TextClassification,
    FaceRecognition,
    ObjectDetection,
]
