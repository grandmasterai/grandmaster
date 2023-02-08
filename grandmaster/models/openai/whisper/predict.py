"""
download the models to ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt  -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt  -P ./weights
"""

from typing import Optional, Any
import torch
import numpy as np
from cog import BasePredictor, Input, Path, BaseModel

from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import format_timestamp

from grandmaster.Output import Transcription


class Predictor(BasePredictor):
    def setup(self):
        self.models = {}
        for model in ["tiny", "base", "small", "medium", "large-v1", "large-v2"]:
            with open(f"weights/{model}.pt", "rb") as fp:
                checkpoint = torch.load(fp, map_location="cpu")
                dims = ModelDimensions(**checkpoint["dims"])
                self.models[model] = Whisper(dims)
                self.models[model].load_state_dict(checkpoint["model_state_dict"])

    def predict(
        self,
        tasks: {
            "transcribe": {
                "inputs": [],
                "outputs": [],
            }
        },
        audio: Path = Input(description="Audio file"),
        model: str = Input(
            default="base",
            choices=["tiny", "base", "small", "medium", "large-v1", "large-v2"],
            description="Choose a Whisper model.",
        ),
        from_language: str = Input(
            choices=list(
                sorted(LANGUAGES.keys())
                + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()])
            ),
            default=None,
            description="language spoken in the audio, specify None to perform language detection",
        ),
    ) -> Transcription:
        print(f"Transcribe with {model} model")
        m = self.models[model].to("cuda")

        temperature = 0
        patience = None
        suppress_tokens = "-1"
        initial_prompt = None
        condition_on_previous_text = True
        temperature_increment_on_fallback = 0.2
        compression_ratio_threshold = 2.4
        logprob_threshold = -1.0
        no_speech_threshold = 0.6

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        args = {
            "language": from_language,
            "patience": patience,
            "suppress_tokens": suppress_tokens,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
        }

        result = m.transcribe(str(audio), temperature=temperature, **args)

        transcription = write_vtt(result["segments"])

        # TODO
        translate = False
        if translate:
            translation = m.transcribe(
                str(audio), task="translate", temperature=temperature, **args
            )["text"]

        return Transcription(
            segments=result["segments"],
            detected_language=LANGUAGES[result["language"]],
            transcription=transcription,
            translation=None,
        )


def write_vtt(transcript):
    result = ""
    for segment in transcript:
        result += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result


def write_srt(transcript):
    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result


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
