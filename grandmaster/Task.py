import inspect
from grandmaster import models
from typing import Optional


class Task:
    def get_model(self, *args, **kwargs) -> models.Model:

        self.task_name = kwargs["task_name"]
        self.model_name = kwargs["model_name"]

        for model in models.all_models:
            if model.task.name == self.task_name:
                return model()
        raise ValueError(f"Task {self.task_name} not found")

    def __call__(self, *args, **kwargs) -> models.Model:
        return self.get_model(*args, **kwargs)


task = Task()
