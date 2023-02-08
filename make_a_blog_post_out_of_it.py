from typing import Union, Literal, List, Any, get_args, Dict

###


T = TypeVar("T")


class Model1(Generic[T]):
    def __init__(self):
        EngineType = Literal["foo", "bar"]
        engine_names: List[EngineType] = list(get_args(EngineType))
        engines: Dict[EngineType, Any] = {}

    def load_engines(self):
        for n in self.engine_names:
            self.engines[n] = n

    def tasks(self):
        return [Task[cls.EngineType](self.engines)]


class Model2:
    def __init__(self):
        self.engine_names = ["v1", "v2"]
        self.engines: Dict[Any, Any] = {}

        for n in self.engine_names:
            self.engines[n] = n

    def tasks(self):
        return [Task[Literal["v1", "v2"]](self.engines)]


from typing import TypeVar, Generic, List, Sequence, Dict


class Task(Generic[T]):
    def __init__(self, engines: Dict[T, Any]):
        self.engines = engines

    def predict(self, engine: T):
        return self.engines[engine]


m1 = Model1()
m1.tasks()[0].predict("large")

m2 = Model2()
m2.tasks()[0].predict("large")

###


class Model1:
    def __init__(self):
        EngineType = Literal["foo", "bar"]
        engine_names: List[EngineType] = list(get_args(EngineType))
        engines: Dict[EngineType, Any] = {}


class Model2:
    EngineType = Literal["foo", "bar"]
    engine_names: List[EngineType] = list(get_args(EngineType))
    engines: Dict[EngineType, Any] = {}


class Model1:
    def __init__(self):
        EngineType = Literal["foo", "bar"]
        self.engine_names: List[EngineType] = list(get_args(EngineType))
        self.engines: Dict[EngineType, Any] = {}
