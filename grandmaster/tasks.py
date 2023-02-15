from typing import List, Generic, TypeVar, Dict, Union, Type, TypedDict
from grandmaster.inputs import File, Image, Label
from grandmaster.outputs import LabelWithScore
from grandmaster.examples import Puppy
from grandmaster.annotations import Input


class ZeroShotImageClassification:
    name = "zero-shot-image-classification"

    class outputsDataTypes(TypedDict):
        labels: List[LabelWithScore]

    def predict(
        self,
        image: File,
        candidate_labels: List[str],
    ) -> outputsDataTypes:
        ...

    examples = [{"image": Puppy, "labels": ["dog", "cat"]}]

    default_model_name = "openai/clip-vit-large-patch14-336"


all_tasks = [ZeroShotImageClassification]
