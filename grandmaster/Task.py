from abc import abstractmethod
from cuid import cuid

from typing import Any, Callable, Dict, Generic, List, Protocol, TypeVar
from grandmaster.Input import parse_query
from grandmaster.proto.mytypes import (
    InputsTypedDict,
    OutputsTypedDict,
    QueryTypedDict,
    RepresentationDC,
    ResultDC,
)

Q = TypeVar("Q", bound=QueryTypedDict)
O = TypeVar("O", bound=QueryTypedDict)
R = TypeVar("R", bound=ResultDC)


class Task(Generic[Q, R, O]):
    id: str
    model_name: str
    inputs: InputsTypedDict
    outputs: OutputsTypedDict

    def __init__(self, inputs, outputs):
        self.id = cuid()
        self.inputs = inputs
        self.outputs = outputs

    def representation(self) -> RepresentationDC:
        ...

    def inference(self, query: Q) -> R:
        x = self.apply(query)
        return self.postprocess(x)

    def apply(self, query: Q) -> O:
        ...

    def preprocess(self, query):
        return parse_query(query, self.inputs)

    def postprocess(self, query) -> R:
        ...

    def check(self, i: InputsTypedDict, o: OutputsTypedDict) -> bool:
        fi: InputsTypedDict = {
            "dataType": i["dataType"],
        }

        fo: OutputsTypedDict = {
            "dataType": o["dataType"],
        }

        orepr = o.get("representation")
        if orepr:
            fo["representation"] = orepr

        return self.inputs == fi and self.outputs == fo

    def form_data_is_valid(self, files, form):
        return True

    # def to_json(self):
    # return {"id": self.id} # return TaskType(id=self.id, model_name=self.t.model_name)
    # ret


def create_task(
    model_name: str,
    inputs: InputsTypedDict,
    outputs: OutputsTypedDict,
    apply: Callable,
    preprocess: Callable,
    postprocess: Callable,
):
    class TaskImpl(Task):
        def __init__(self, model_name, inputs, outputs, apply, preprocess, postprocess):
            super().__init__(inputs, outputs)
            self.model_name = model_name
            self.apply = apply
            self.preprocess = preprocess
            self.postprocess = postprocess

    return TaskImpl(model_name, inputs, outputs, apply, preprocess, postprocess)
