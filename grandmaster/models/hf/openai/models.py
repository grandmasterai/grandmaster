from typing import Any, List
from grandmaster.Model import Model
from grandmaster.Task import create_task
from grandmaster.helper import load_image_from_data
from grandmaster.proto.mytypes import (
    AudioQueryTypedDict,
    ImageZeroShotQueryTypedDict,
    InputsType,
    InputsTypedDict,
    OutputsType,
    OutputsTypedDict,
    ResultLabelDC,
)


class CLIP(Model):
    # image classification
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel

        self.xmodel = lambda: CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.xprocessor = lambda: CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )

        self.model_name = "openai/clip-vit-base-patch16"

        self.inputs: List[InputsTypedDict] = [
            {"dataType": "IMAGE"},
            {"dataType": "TEXT"},
        ]
        self.outputs: List[OutputsTypedDict] = [{"dataType": "LABEL"}]

    def tasks(self):
        return [
            create_task(
                self.model_name,
                self.inputs,
                self.outputs,
                self.apply,
                self.preprocess,
                self.postprocess,
            )
        ]

    def apply(self, query: ImageZeroShotQueryTypedDict) -> Any:
        image = load_image_from_data(query["image"])
        processed = self.xprocessor()(
            text=query["candidate_labels"].split(","),
            images=image,
            return_tensors="pt",
            padding=True,
        )
        x = self.xmodel()(**processed)  # type: ignore
        results = list(x.logits_per_image.softmax(dim=1).detach().numpy()[0])
        out: List[ResultLabelDC] = []
        for i, r in zip(query["candidate_labels"].split(","), results):
            out.append(ResultLabelDC(label=i, score=r.item()))
        return out

    def postprocess(self, x: Any) -> List[ResultLabelDC]:
        return x

    def preprocess(self, query: ImageZeroShotQueryTypedDict) -> Any:
        return query


"""
class Whisper(Model):
    def __init__(self):
        from whisper import load_model
        self.model_name = "openai/whisper-base"
        self.modelx = load_model("base")
        
    def model(self, query : AudioQueryTypedDict) -> :
        # TODO
        # query["audio"]
        result = self.modelx.transcribe() # type: ignore
        text : str = result["text"] # type: ignore
        return ResultsDataClass(results=ResultText(text=text))
"""
