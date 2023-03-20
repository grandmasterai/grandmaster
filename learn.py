from grandmaster.tasks import all_tasks
from pydantic import BaseModel, create_model
from pprint import pprint
from typing import List, Literal, Union

"""
for task in all_tasks:
    InputsWithModel = task.create_inputs_with_models()
    pprint(InputsWithModel.schema())
"""
