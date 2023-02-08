from typing import List

from grandmaster.models import all_models
from grandmaster.proto.mytypes import TaskTypedDict


# def get_task(slug: str):
#    for m in all_models:
#        t = m().get_task(inputs, outputs)
#        if t:
#            return t
#    print("Task not found")
#    return None


def list_tasks() -> List[TaskTypedDict]:
    tasks = []
    for m in all_models:
        for t in m().tasks():
            tasks.append({"inputs": t.inputs, "outputs": t.outputs})
    return tasks


def list_models(task: TaskTypedDict):
    models = []
    for m in all_models:
        m = m()
        for t in m.tasks():
            if t.inputs == task["inputs"] and t.outputs == task["outputs"]:
                models.append(
                    {
                        "inputs": t.inputs,
                        "outputs": t.outputs,
                        "settings": m.settings,
                        "model_name": m.model_name,
                    }
                )
    return models


def get_model(inputs, outputs, settings):
    pass
