from pathlib import Path
from typing import List, Optional
from todo.mytypes import InputsTypedDict, OutputsTypedDict

from glob import glob

from grandmaster.config import EXAMPLES_DIR


class Example:
    pass


def get_examples(
    inputs: List[InputsTypedDict], outputs: Optional[List[OutputsTypedDict]] = None
):
    examples = list((EXAMPLES_DIR / inputs[0]["dataType"].lower()).glob("*"))
    examples = [e for e in examples if e.name != ".DS_Store"]
    return examples


def get_examples_url(
    inputs: List[InputsTypedDict], outputs: Optional[List[OutputsTypedDict]] = None
):

    return [example_path_to_url(e) for e in get_examples(inputs, outputs)]


def example_path_to_url(path: Path):
    BASE = "http://localhost:3000"
    parts = path.parts
    return f"{BASE}/ex/{parts[-1]}"
