from grandmaster import get_task

task = get_task(task_name="ZERO_SHOT_IMAGE_CLASSIFICATION", model_name="whatever")  # type: ignore
task.predict(
    query={
        "image": "https://i.imgur.com/7YmK0bP.jpg",
        "candidate_labels": [{"label": "cat"}, {"label": "dog"}],
    },
    model_settings={"x": "y"},
)

# pro: generic
# cons:
#  - type format not explicit from task_name
#  - need to have a task name for each task (that in any case)
# ------------------------------------------------------------------------------------------------------------------------

from grandmaster import get_model  # type: ignore

model = get_model(task_name="ZERO_SHOT_IMAGE_CLASSIFICATION", model_name="whatever")  # type: ignore
model.predict(
    query={
        "image": "https://i.imgur.com/7YmK0bP.jpg",
        "candidate_labels": [{"label": "cat"}, {"label": "dog"}],
    },
    model_settings={"x": "y"},
)

# ------------------------------------------------------------------------------------------------------------------------

from grandmaster import get_model  # type: ignore
from grandmaster.inputs import InputImage, InputLabel
from grandmaster.outputs import OutputLabel

task = get_task(inputs=[InputImage], outputs=[InputLabel], task_name="AUTO")  # type: ignore

task.list_models()
model = task.load_model(model_name="AUTO")

model.predict(
    query={
        "image": InputImage.open("https://i.imgur.com/7YmK0bP.jpg")),
        "candidate_labels": [OutputLabel(name="name")],
    },
    model_settings={}
)

# ------------------------------------------------------------------------------------------------------------------------

import grandmaster

grandmaster.api_key = os.getenv("GRANDMASTER_API_KEY")

task = grandmaster.get_task(inputs={{inputs}}, outputs={{ outputs }}, model = "AUTO" {{ cloud }})
task.inference()

# ------------------------------------------------------------------------------------------------------------------------


