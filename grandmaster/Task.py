from abc import abstractmethod
from cuid import cuid

from typing import Any, Callable, Dict, Generic, List, Protocol, TypeVar
from grandmaster.proto.mytypes import (
    ImageZeroShotQueryTypedDict,
    InputsTypedDict,
    OutputsTypedDict,
    QueryType,
    QueryTypedDict,
    RepresentationDC,
    ResultDC,
    TextZeroShotQueryTypedDict,
)

Q = TypeVar("Q", bound=QueryTypedDict)
O = TypeVar("O", bound=QueryTypedDict)
R = TypeVar("R", bound=ResultDC)


class Task(Generic[Q, R, O]):
    id: str

    def __init__(self, inputs, outputs):
        self.id = cuid()
        self.inputs = inputs
        self.outputs = outputs

    def representation(self) -> RepresentationDC:
        ...

    def inference(self, query: Q) -> R:
        x = self.model(query)
        return self.postprocess(x)

    def model(self, query: Q) -> O:
        ...

    def postprocess(self, x) -> R:
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


def create_task(inputs: InputsTypedDict, outputs: OutputsTypedDict, model, postprocess):
    class TaskImpl(Task):
        def __init__(self, inputs, outputs):
            self.model = model
            self.postprocess = postprocess
            self.inputs = inputs
            self.outputs = outputs

    return TaskImpl(inputs, outputs)
