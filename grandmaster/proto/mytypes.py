from dataclasses import dataclass

from typing import List, Optional, Type, Union

from typing import TypedDict, Literal

from typing_extensions import NotRequired

# inputs
class InputsTypedDict(TypedDict):
    dataType: Literal["TEXT", "IMAGE", "VIDEO", "AUDIO", "PROMPT"]


# outputs
class OutputsTypedDict(TypedDict):
    dataType: Literal["BOUNDINGBOX", "LABEL", "COMPLETION"]
    representation: NotRequired[Literal["FACE", "PLATE"]]


# model
class ModelTypedDict(TypedDict):
    name: str  # e.g. "bert-base-uncased"
    domain: NotRequired[Literal["SCIENTIFIC"]]


# query
class ImageQueryTypedDict(TypedDict):
    image: bytes


class ImageZeroShotQueryTypedDict(TypedDict):
    image: bytes
    candidate_labels: str  # comma separated


class TextZeroShotQueryTypedDict(TypedDict):
    text: str
    candidate_labels: str  # comma separated


class AudioQueryTypedDict(TypedDict):
    audio: bytes


class PromptQueryTypedDict(TypedDict):
    text: str


# results
# pascal_voc style


@dataclass
class ResultBoundingBoxBoxDC:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class ResultBoundingBoxDC:
    box: ResultBoundingBoxBoxDC
    score: Optional[float]
    label: str


@dataclass
class ResultBoundingBoxOfACarDC:
    box: ResultBoundingBoxBoxDC
    score: float
    label: str


@dataclass
class ResultLabelDC:
    label: str
    score: float


@dataclass
class ResultTextDC:
    text: str


@dataclass
class ResultGeneratedTextDC:
    text: str


ResultDC = Union[List[ResultLabelDC], List[ResultBoundingBoxDC]]
ResultType = Union[Type[List[ResultLabelDC]], Type[List[ResultBoundingBoxDC]]]


@dataclass
class ResultsDC:
    results: List[ResultDC]


# representation output


@dataclass
class RepresentationDC:
    output: Literal["PLATE"]


# visibles

InputsType = Type[InputsTypedDict]
OutputsType = Type[OutputsTypedDict]

QueryTypedDict = Union[
    ImageQueryTypedDict,
    ImageZeroShotQueryTypedDict,
    TextZeroShotQueryTypedDict,
    AudioQueryTypedDict,
]

QueryType = Union[
    Type[ImageQueryTypedDict],
    Type[ImageZeroShotQueryTypedDict],
    Type[TextZeroShotQueryTypedDict],
    Type[AudioQueryTypedDict],
]
