from typing import Any, Optional, Union
from grandmaster.helper import load_image_from_url

from grandmaster.proto.mytypes import InputsTypedDict, QueryTypedDict


class Input:
    def __init__(self):
        self.type = "image"


class Image:
    def __init__(self, representation: Optional[str] = None):
        self.name = "image"
        self.representation = representation


class Text:
    def __init__(self):
        self.name = "text"


class Prompt:
    def __init__(self):
        self.name = "prompt"


from pathlib import Path


def load_image(image: Union[Path, str]):
    pass


def parse_query(query: Any, inputs: InputsTypedDict) -> QueryTypedDict:
    if inputs["dataType"] == "IMAGE":
        if type(query) == str:
            if "https://" in query or "http://" in query:
                return load_image_from_url(query)
    raise ValueError("query type not supported")


from pydantic import BaseModel


class Audio(BaseModel):
    file: Any  #
    language: Optional[str] = None
