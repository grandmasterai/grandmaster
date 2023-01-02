from inspect import signature
from typing import Callable, Dict, Optional, Protocol
from abc import abstractmethod

from typing import List

from grandmaster.Task import Task
from grandmaster.proto.mytypes import ImageZeroShotQueryTypedDict, InputsTypedDict, OutputsTypedDict, QueryType, QueryTypedDict, ResultDC, ResultType, TextZeroShotQueryTypedDict

from inspect import signature, _empty

class Model(Protocol):

    model_name : str

    @abstractmethod
    def tasks(self) -> List[Task]:
        ...

    def get_task_signature(self, inputs : InputsTypedDict, outputs : OutputsTypedDict) -> Optional[Task]:
        # outdated
        for t in self.tasks():
            query = signature(t.inference).parameters["query"].annotation
            result = signature(t.inference).return_annotation
            if query == _empty or result == _empty:
                raise ValueError("query or result type is not set!")
            if (queryTypeToInputs(query) == inputs and resultToOutputs(result) == outputs):
                return t
        return None
    
    def get_task(self, inputs : InputsTypedDict, outputs : OutputsTypedDict) -> Optional[Task]:
        for t in self.tasks():
            if t.check(inputs, outputs):
                return t
        return None

def queryTypeToInputs(queryType : QueryType) -> Optional[InputsTypedDict]:

    d : Dict[QueryType, InputsTypedDict] = {
        TextZeroShotQueryTypedDict : {"dataType": "TEXT"},
        ImageZeroShotQueryTypedDict : {"dataType": "IMAGE"},
    }
    return d.get(queryType)

def resultToOutputs(resultType : ResultType) -> Optional[OutputsTypedDict]:
    d : Dict[ResultType, OutputsTypedDict] = {
        resultType : {"dataType": "LABEL"},
    }
    return d.get(resultType)