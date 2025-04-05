
from huggingface_hub import hf_hub_download

pt_path = hf_hub_download(
    repo_id="jane-street/2025-03-10",
    filename="model_3_11.pt",
)

import pickle

with open(pt_path, "rb") as f:
    data = pickle.load(f)

# %%
import cloudpickle
import io
import pickle
__globals__ = {"a": "foo"}

class Pickler(cloudpickle.CloudPickler):
    @staticmethod
    def persistent_id(obj):
        if id(obj) == id(__globals__):
            return "__globals__"

class Unpickler(pickle.Unpickler):
    @staticmethod
    def persistent_load(pid):
        return {"__globals__": __globals__}[pid]

get = eval('lambda: a', __globals__)
file = io.BytesIO()
Pickler(file).dump(get)
dumped = file.getvalue()
assert b'foo' not in dumped
get = Unpickler(io.BytesIO(dumped)).load()
assert id(__globals__) == id(get.__globals__)
assert 'foo' == get()
__globals__['a'] = 'bar'
assert 'bar' == get()

# %%
