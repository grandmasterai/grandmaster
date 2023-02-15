# CONTRIBUTING

**Design principles**

- Seamless integration with the API
- Both the API and the SDK accepts and returns the same types (JSON dictionaries, "typed").
- Zero-config mantra. We try to keep the models as simple as possible. Our vision is that in 2/3 years from now foundations models will be enough good.
  - If a model requires too many parameters to work, probably is not a good fit for Grandmaster.

**Canonical type**

- Preferably, every input or output should have only one canonical type representation.
- For instance, transcriptions can be represented in several ways and formats (.txt, .srt, .vtt). To keep things simple, we return only one type (in this case, VTT, as it is the most generic one that contains more metadata!). We can eventually explain the customer how to view that type in a different format.


### Add a new model

- Add the model under `models/`. See other models for an example.


### Add a new task

- Add the tasks under `tasks.py`. If you can, use existing types. 
