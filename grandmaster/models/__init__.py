from typing import List, Type
from grandmaster.Model import Model

# from grandmaster.models.hf.facebook.models import BART, Galactica, Detr
# from grandmaster.models.hf.openai.models import CLIP, Whisper


# from grandmaster.models.hf.facebook.models import BART
# from grandmaster.models.hf.openai.models import CLIP

# from grandmaster.models.hf.facebook.models import Detr
# from grandmaster.models.community.models import FaceRecognition, YOLOPlate

from grandmaster.models.openai.clip.predict import CLIP

all_models = [CLIP]  # , YOLOPlate, , Galactica, Whisper, ]
