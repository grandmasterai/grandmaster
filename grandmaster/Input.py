from typing import Optional


class Input:
    def __init__(self):
        self.type = "image"


class Image:
    def __init__(self, representation: Optional[str] = None):
        self.name = "image"
        self.representation = representation


class Text:
    def __init__(self):
        self.name = "text"


class Prompt:
    def __init__(self):
        self.name = "prompt"


class Audio:
    def __init__(self, language: Optional[str] = None):
        self.name = "audio"
        self.language = language
