# %%
from textwrap import dedent

import torch as t
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

pt_path = hf_hub_download(
    repo_id="jane-street/2025-03-10",
    filename="model_3_11.pt",
)
seq = t.load(pt_path, weights_only=False)
# %%


class Input(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = t.nn.Parameter(
            t.randint(ord("A"), ord("Z"), (55,), requires_grad=True, dtype=t.float32)
        )

    def forward(self, seq):
        x = self.inp
        for layer in seq:
            x = layer(x)
        return x


inp = Input()

# %%

learning_rate = 0.1
num_steps = 1000
optimizer = t.optim.Adam(inp.parameters(), lr=learning_rate)

pbar = tqdm(range(num_steps))
for i in pbar:
    optimizer.zero_grad()

    outp = inp(seq)
    loss = -outp.sum()
    loss.backward()
    optimizer.step()

    if i == 0:
        print("\nChecking model differentiability:")
        print(f"Input requires grad: {inp.inp.requires_grad}")
        print(f"Gradients exist: {inp.inp.grad is not None}")
        print(f"Model output requires grad: {outp.requires_grad}")

    if i % 10 == 0:
        inp_range = inp.inp.max().item() - inp.inp.min().item()
        inp_grad_range = inp.inp.grad.max().item() - inp.inp.grad.min().item()
        inp_grad_min = inp.inp.grad.min().item()
        inp_grad_max = inp.inp.grad.max().item()
        inp_grad_norm = inp.inp.grad.norm().item()
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


# Get the final optimized input
optimized_input = inp.inp.detach()
print("\nFinal optimized input:")
print(optimized_input)
print(f"\nFinal output value: {seq(optimized_input).item():.4f}")


# %%
