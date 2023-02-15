from abc import ABC


class Model(ABC):
    name = None

    def prediction(self, *args, **kwargs):
        raise NotImplementedError
