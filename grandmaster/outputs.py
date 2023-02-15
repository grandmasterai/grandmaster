from typing import Any, Optional
from pydantic import BaseModel
from pathlib import Path

from whisper.tokenizer import LANGUAGES  # TODO make our own


class LabelWithScore(BaseModel):
    label: str
    score: float


class Transcription(BaseModel):
    detected_language: str
    transcription: str
    segments: Any  # TODO make it more specific
    translation: Optional[str]
    txt_file: Optional[Path]  # TODO use File
    srt_file: Optional[Path]  # TODO use one of the two.
