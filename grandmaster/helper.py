import io
from typing import Any

from PIL import Image
import requests

import datasets

from grandmaster.proto.mytypes import ImageZeroShotQueryTypedDict


def load_image_data_from_url(url: str) -> bytes:
    # same as equests.get(url).content
    return requests.get(url, stream=True).raw.read()


def load_image_from_data(data):
    image = Image.open(io.BytesIO(data))
    return image


def get_audio():
    ds = load_dataset("common_voice", "fr", split="test", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    input_speech = next(iter(ds))["audio"]["array"]
    return input_speech
