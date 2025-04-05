# %%
import matplotlib.pyplot as plt
import torch as t
from huggingface_hub import hf_hub_download
from nnsight import NNsight

pt_path = hf_hub_download(
    repo_id="jane-street/2025-03-10",
    filename="model_3_11.pt",
)
seq = t.load(pt_path, weights_only=False)

class Wrapper(t.nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.seq = seq

    def forward(self, x):
        return self.seq(x)

model = NNsight(Wrapper(seq))

# %%

third_last = model.seq[5436]
second_last = model.seq[5438]
last = model.seq[5440]
# %%

pos = t.where(last.weight.data > 0, last.weight.data, 0)
neg = t.where(last.weight.data < 0, last.weight.data, 0)

print(f"""
Final Linear Layer ({last.weight.data.shape})

# Weights

{last.weight.data}
pos: {pos.sum()}, neg: {neg.sum()}

# Biases

{last.bias.data}

""")

plt.imshow(last.weight.data)
plt.axis("off")
plt.show()

# %%

pos = t.where(second_last.weight.data > 0, second_last.weight.data, 0)
neg = t.where(second_last.weight.data < 0, second_last.weight.data, 0)

plt.imshow(second_last.weight.data)
plt.axis("off")
plt.show()

print(f"""
Second Last Layer ({second_last.weight.data.shape})

# Weights

pos: {pos.sum()}, neg: {neg.sum()}

First 20 weights, (has positive birst)
{second_last.weight[0, :20]}

Last 20 weights, (has negative bias)
{second_last.weight[-1, -20:]}

# Biases

{second_last.bias.data}
""")

# %%

plt.imshow(third_last.weight.data)
plt.axis("off")
plt.show()


# %%
# %%
with model.trace("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") as tracer:
    i = model.input.save()
    first_inp = model.seq[0].input.save()
    first_acts = model.seq[0].output.save()
    output = model.output.save()
    second_last_acts = second_last.output.save()
    last_acts = last.output.save()

print(output)
print(second_last_acts)
print(last_acts)

# %%

print(first_inp)
# %%

print(first_acts)
