from pathlib import Path
from typing import Any, List, Optional


class Labels:
    def __init__(self, values: List[str]):
        self.name = "labels"
        self.values = values


class BoundingBox:
    def __init__(self, representation: Optional[str] = None):
        self.name = "boundingbox"
        self.representation = representation


class Generated:
    def __init__(self):
        self.name = "generated"


class Text:
    def __init__(self, language: Optional[str] = None):
        self.name = "text"
        self.language = language


from whisper.tokenizer import LANGUAGES  # TODO make our own

from pydantic import BaseModel


class Transcription(BaseModel):
    detected_language: str
    transcription: str
    segments: Any  # TODO make it more specific
    translation: Optional[str]
    txt_file: Optional[Path]  # TODO returns directly
    srt_file: Optional[Path]  # TODO returns directly
