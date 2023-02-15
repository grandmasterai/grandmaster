from inspect import signature
from grandmaster.models import all_models


def validate():
    for model in all_models:
        if signature(model.predict) != signature(model.task.predict):
            print(signature(model.predict))
            print(signature(model.task.predict))
            print("\n")
