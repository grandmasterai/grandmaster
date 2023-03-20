from grandmaster.Task import predict

from enum import Enum


class TASK(str, Enum):
    ZERO_SHOT_IMAGE_CLASSIFICATION = "ZERO_SHOT_IMAGE_CLASSIFICATION"


TaskToName = {TASK.ZERO_SHOT_IMAGE_CLASSIFICATION: "openai/clip-vit-base-patch16"}
