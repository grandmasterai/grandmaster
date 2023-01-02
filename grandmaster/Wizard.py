from typing import Optional
from grandmaster.Task import Task

from grandmaster.models import all_models
from grandmaster.proto.mytypes import InputsTypedDict, OutputsTypedDict

def get_task(inputs : InputsTypedDict, outputs : OutputsTypedDict) -> Optional[Task]:
    for m in all_models:
        t = m().get_task(inputs, outputs)
        if t:
            return t

    print("Task not found")
    return None