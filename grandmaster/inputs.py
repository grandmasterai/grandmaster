from pydantic import BaseModel
from fastapi import UploadFile

File = UploadFile


class Image(BaseModel):
    image: bytes


class Audio(BaseModel):
    audio: bytes


class Label(BaseModel):
    label: str
