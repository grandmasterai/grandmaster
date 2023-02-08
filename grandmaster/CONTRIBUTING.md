
- Model

model.py

    class Name(Model):
        def tasks() -> List[Task]
    

    class Task
        self.inputsType = List[TaskInputType]
        self.outputsType = List[TaskOutputType]
        self.settings = Dict[str, str]
        self.name = TASK_NAME

        self.load_model(): # load model in memory
        self.predict(TaskQuery) -> List[TaskOutput]
