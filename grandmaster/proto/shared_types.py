# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: shared_types.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import List

import betterproto


class InputDataType(betterproto.Enum):
    IMAGE = 0
    VIDEO = 1
    AUDIO = 2
    TEXT = 3
    TEST = 4


class InputRepresentation(betterproto.Enum):
    CAR = 0


class OutputDataType(betterproto.Enum):
    BOUNDINGBOX = 0
    LABEL = 1


class OutputRepresentation(betterproto.Enum):
    FACE = 0


@dataclass
class Query(betterproto.Message):
    audio: "QueryAudio" = betterproto.message_field(1, group="q")
    image: "QueryImage" = betterproto.message_field(2, group="q")
    video: "QueryVideo" = betterproto.message_field(3, group="q")
    text: "QueryText" = betterproto.message_field(4, group="q")


@dataclass
class QueryAudio(betterproto.Message):
    audio: bytes = betterproto.bytes_field(1)


@dataclass
class QueryImage(betterproto.Message):
    image: bytes = betterproto.bytes_field(1)


@dataclass
class QueryVideo(betterproto.Message):
    video: bytes = betterproto.bytes_field(1)


@dataclass
class QueryText(betterproto.Message):
    text: str = betterproto.string_field(1)


@dataclass
class Result(betterproto.Message):
    label: "ResultLabel" = betterproto.message_field(1, group="r")
    boundingbox: "ResultBoundingBox" = betterproto.message_field(2, group="r")
    other: str = betterproto.string_field(3)


@dataclass
class ResultLabel(betterproto.Message):
    label: str = betterproto.string_field(1)
    score: float = betterproto.float_field(2)


@dataclass
class ResultBoundingBoxBox(betterproto.Message):
    xmin: float = betterproto.float_field(1)
    xmax: float = betterproto.float_field(2)
    ymin: float = betterproto.float_field(3)
    ymax: float = betterproto.float_field(4)


@dataclass
class ResultBoundingBox(betterproto.Message):
    box: "ResultBoundingBoxBox" = betterproto.message_field(1)
    label: str = betterproto.string_field(2)
    score: float = betterproto.float_field(3)


@dataclass
class ResultSpeechRecognitionChunk(betterproto.Message):
    text: str = betterproto.string_field(1)
    timestamps: List[float] = betterproto.float_field(2)


@dataclass
class ResultSpeechRecognition(betterproto.Message):
    text: str = betterproto.string_field(1)


@dataclass
class ResultMask(betterproto.Message):
    score: float = betterproto.float_field(1)
    label: str = betterproto.string_field(2)


@dataclass
class ResultTokenMask(betterproto.Message):
    score: float = betterproto.float_field(1)
    token: str = betterproto.string_field(2)


@dataclass
class ResultTokenNER(betterproto.Message):
    group: str = betterproto.string_field(1)
    score: float = betterproto.float_field(2)
    word: str = betterproto.string_field(3)
    start: int = betterproto.int32_field(4)
    end: int = betterproto.int32_field(5)


@dataclass
class ResultAnswer(betterproto.Message):
    score: float = betterproto.float_field(1)
    start: int = betterproto.int32_field(2)
    end: int = betterproto.int32_field(3)
    answer: str = betterproto.string_field(4)


@dataclass
class ResultTableAnswer(betterproto.Message):
    answer: str = betterproto.string_field(1)


@dataclass
class Input(betterproto.Message):
    data_type: "InputDataType" = betterproto.enum_field(1)
    representation: "InputRepresentation" = betterproto.enum_field(2)


@dataclass
class Output(betterproto.Message):
    data_type: "OutputDataType" = betterproto.enum_field(1)
    representation: "OutputRepresentation" = betterproto.enum_field(2)


@dataclass
class Task(betterproto.Message):
    id: str = betterproto.string_field(1)
    model_name: str = betterproto.string_field(2)
