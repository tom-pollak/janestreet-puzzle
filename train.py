"""
Gradient works, but because we're basically dealing with dead relus we get 0 gradient back to our input.
"""
# %%
from textwrap import dedent

import torch as t
from huggingface_hub import hf_hub_download
from nnsight import NNsight
from tqdm.auto import tqdm

pt_path = hf_hub_download(
    repo_id="jane-street/2025-03-10",
    filename="model_3_11.pt",
)


class Model(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = t.load(pt_path, weights_only=False)

    def forward(self, inp):
        x = inp
        for layer in self.seq:
            x = layer(x)
        return x


model = Model()
sight = NNsight(model)

inp = t.nn.Parameter(
    t.randint(ord("A"), ord("z"), (55,), requires_grad=True, dtype=t.float32)
)

# %%

learning_rate = 0.1
num_steps = 1000
optimizer = t.optim.Adam([inp], lr=learning_rate)

# %%

pbar = tqdm(range(num_steps))
for i in pbar:
    optimizer.zero_grad()

    # acts = model(inp)
    # loss = -acts.mean()
    # loss.backward()
    # inp_grad = inp.grad

    with sight.trace(inp) as tracer:
        acts = sight.seq[5440].output.save()
        loss = -acts.mean()
        loss.save()
        loss.backward()

    inp_grad = inp.grad

    if i % 10 == 0:
        inp_range = inp.max().item() - inp.min().item()
        inp_grad_min = inp_grad.min().item()
        inp_grad_max = inp_grad.max().item()
        inp_grad_range = inp_grad_max - inp_grad_min
        inp_grad_norm = inp_grad.norm().item()
        print(f"Loss: {loss.item():.4f}")
        print(
            dedent(
                f"""\
                Input range: {inp_range:.4f}
                Input grad range: {inp_grad_range:.4f}
                Input grad min: {inp_grad_min:.4f}
                Input grad max: {inp_grad_max:.4f}
                Input grad norm: {inp_grad_norm:.4f}
                """
            )
        )

    optimizer.step()

# Get the final optimized input
optimized_input = inp.detach()
print("\nFinal optimized input:")
print(optimized_input)
print(f"\nFinal output value: {model(optimized_input).item():.4f}")


# %%
