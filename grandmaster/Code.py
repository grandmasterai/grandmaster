import re

import jinja2
from typing import Literal
from pathlib import Path

from grandmaster.proto.mytypes import InputsTypedDict, OutputsTypedDict

from black import format_str, FileMode


def get_template(template: Literal["python", "curl"]):
    file = (
        Path(__file__).resolve().parent / "templates" / f"{template}.py.jinja2"
    ).open()
    return jinja2.Template(file.read())


class Code:
    def __init__(self, inputs: InputsTypedDict, outputs: OutputsTypedDict):
        self.inputs = {k: v for k, v in inputs.items() if v is not None}
        self.outputs = {k: v for k, v in outputs.items() if v is not None}

    def get_python(self):
        template = get_template("python")

        x = template.render(inputs=dict(self.inputs), outputs=dict(self.outputs))
        x = re.sub(r"\n\s+", "\n\n", x)
        return format_str(x, mode=FileMode())

    def get_curl(self):
        return "curl code"

    def to_json(self):
        return {"python": self.get_python(), "curl": self.get_curl()}


def get_code(inputs: InputsTypedDict, outputs: OutputsTypedDict):
    return Code(inputs, outputs).to_json()
