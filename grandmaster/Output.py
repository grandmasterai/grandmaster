from typing import List, Optional


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
