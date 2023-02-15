import re
import jinja2

from typing import Any, Dict, List, Literal, Union
from pathlib import Path

from grandmaster.Example import get_examples_url
from grandmaster.proto.mytypes import InputsTypedDict, OutputsTypedDict

from black import format_str, FileMode

"""
def get_template(template: Literal["python", "curl"]):
    file = (
        Path(__file__).resolve().parent / "templates" / f"{template}.py.jinja2"
    ).open()
    return jinja2.Template(file.read())

class Code:
    def __init__(
        self,
        inputs: List[InputsTypedDict],
        outputs: List[OutputsTypedDict],
        isCloud: bool,
    ):

        self.inputs = [self.clean_param(i) for i in inputs]
        self.outputs = [self.clean_param(o) for o in outputs]
        self.isCloud = isCloud

    @staticmethod
    def clean_param(param: Any):
        out = {}
        if param.get("dataType") is not None:
            out["dataType"] = param["dataType"]
        if param.get("representation") is not None:
            out["representation"] = param["representation"]
        if param.get("of") is not None:
            out["representation"] = param["representation"]

        return out

    def get_python(self):
        template = get_template("python")

        x = template.render(
            inputs=self.inputs,
            outputs=self.outputs,
            examples=get_examples_url(self.inputs, self.outputs),
            isCloud=self.isCloud,
        )
        x = re.sub(r"\n\s+", "\n\n", x)
        return format_str(x, mode=FileMode(line_length=55))  # 88 default

    def get_curl(self):
        return "curl code"

    def to_json(self):
        return {"python": self.get_python(), "curl": self.get_curl()}


def get_code(
    inputs: List[InputsTypedDict], outputs: List[OutputsTypedDict], isCloud: bool
):
    return Code(inputs, outputs, isCloud).to_json()
"""


def get_samples(task):
    return [
        {
            "lang": "Python",
            "source": "python",
        },
        {
            "lang": "CURL",
            "source": "...",
        },
    ]
