# CONTRIBUTING

**Design principles**

- Seamless integration with the API
- Both the API and the SDK accepts and returns the same types (JSON dictionaries, "typed").
- Zero-config mantra. We try to keep the models as simple as possible. Our vision is that in 2/3 years from now foundations models will be enough good.
- If a model requires too many parameters to work, probably is not a good fit for Grandmaster.
- Documentation is generated from the API

**Canonical type**

- Preferably, every input or output should have only one canonical type representation.
- For instance, transcriptions can be represented in several ways and formats (.txt, .srt, .vtt). To keep things simple, we return only one type (in this case, VTT, as it is the most generic one that contains more metadata!). We can eventually explain the customer how to view that type in a different format.


### Add a new model

- Add the model under `models/`. See other models for an example.


### Add a new task

- Add the tasks under `tasks.py`. If you can, use existing types. 


### Rules 

- Every class should inherit from pydantic.BaseClass (i.e use pydantic for classes everywhere!)

### High-level

The library is composed of the following main **concepts**:

- **Models**. They are found under the folder `models`, organized in folders like in HF (company/model_name).
    Each "leaf" represents a ModelCard. Each model instance can be associated with multiple tasks, that are "exported" (ModelForTask). Models input and outputs are constrained by the associated task.

- **Tasks**. They are defined in one place as this help keep consistency. Each task define the supported input and output types.


- **Types**. Tasks input and outputs parameters are defined in a single place. This help keep consistency.

### Open questions:

- Would it be better if each _exported_ class in the `models` folder would inherit from Task instead of model or maybe antoher class called `ModelForTask` or something like that?