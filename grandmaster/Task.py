import inspect
from grandmaster import models
from typing import Optional, List, Literal

from pydantic import BaseModel, create_model


class Task:
    def get_model(self, input) -> models.Model:

        self.task_name = input.task_name
        self.model_name = input.model_name

        for model in models.all_models:
            if model.task.name == self.task_name:
                return model()
        raise ValueError(f"Task {self.task_name} not found")

    def __call__(self, *args, **kwargs) -> models.Model:
        return self.get_model(*args, **kwargs)


from pydantic import BaseModel

# TODO generate this dynamically
class Input(BaseModel):
    task_name: str
    model_name: str


def predict(input: Input):
    task = Task()
    model = task.get_model(input)
    return model.prediction(input)


# task = Task()
