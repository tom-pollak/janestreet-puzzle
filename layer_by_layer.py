# %%

import torch as t
from huggingface_hub import hf_hub_download
from nnsight import NNsight
from tqdm.auto import tqdm

device = t.device("mps")
t.set_default_device(device)

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


model = Model().to(device)
sight = NNsight(model)

rand_toks = t.nn.Parameter(
    t.randint(ord("A"), ord("z"), (55,), requires_grad=True, dtype=t.float32)
)

start_idxs = [i for i, l in enumerate(model.seq) if isinstance(l, t.nn.ReLU)][::-1]
final_idx = start_idxs.pop(0) - 1  # final linear layer -- optimize to be 1.0
start_idxs.append(0)

# %%

min_lr = 1e-3
max_lr = 5e-3
eps_mse = 1e-6
eps_cossim = 5e-3
cos_alpha = 0.0  # 1e-3
weight_decay = 0.0  # 1e-3
batch_size = 100
# %%


def lr_scheduler(step, n_steps, initial_lr, warmup_steps, final_lr):
    if step < warmup_steps:
        return initial_lr + (1e-1 - initial_lr) * (step / warmup_steps)
    elif step < n_steps:
        return 1e-1 - (1e-1 - final_lr) * (
            (step - warmup_steps) / (n_steps - warmup_steps)
        )
    else:
        return final_lr


def optimize_input(
    input_idx,
    output_idx,
    target,
    inp_weights=None,
    min_steps=100,
    max_steps=1000,
    inp_stddev=1.0,
):
    input_layer = model.seq[input_idx]
    if not hasattr(input_layer, "out_features"):  # isinstance(input_layer, t.nn.ReLU):
        inp_dim = model.seq[input_idx - 1].out_features
    else:
        inp_dim = input_layer.in_features

    if inp_weights is None:
        inp_weights = (
            t.randn(batch_size, inp_dim, requires_grad=True, dtype=t.float32)
            * inp_stddev
        )
        # layer_W = model.seq[input_idx + 1].weight.data
        # layer_b = model.seq[input_idx + 1].bias.data
        # layer_W_norm = (layer_W.max() - layer_W.min()) / 2
        # layer_W.div_(layer_W_norm)
        # layer_b.div_(layer_W_norm)

    else:
        assert inp_weights.shape == (
            batch_size,
            inp_dim,
        )
        inp_weights.requires_grad = True
        inp_weights.grad = None

    if target.ndim == 1:
        target = target.unsqueeze(0).expand(batch_size, -1)

    inp = t.nn.Parameter(inp_weights)
    warmup_steps = min_steps // 10
    optimizer = t.optim.Adam([inp], lr=min_lr, weight_decay=weight_decay)

    pbar = tqdm(range(max_steps))
    for i in pbar:
        current_lr = lr_scheduler(i, min_steps, min_lr, warmup_steps, max_lr)
        optimizer.param_groups[0]["lr"] = current_lr

        optimizer.zero_grad()
        sub_model = model.seq[input_idx:output_idx+1]
        acts = sub_model(inp)
        loss = t.nn.functional.mse_loss(acts, target)

        # zinp = t.nn.functional.normalize(inp, p=2, dim=-1)
        # bottom_tri = t.tril(t.ones_like(zinp @ zinp.T), diagonal=-1)
        # cossim = (zinp @ zinp.T) * bottom_tri
        # cossim_loss = t.nn.functional.mse_loss(cossim, bottom_tri)

        loss.backward()
        optimizer.step()

        grad_norm = inp.grad.norm()
        pbar.set_postfix(
            loss=loss.item(),
            grad_norm=grad_norm.item(),
        )

        if loss < eps_mse and i > min_steps:
            break

    pbar.close()
    return inp.detach()


# %%

# need weight decay for this one
weight_decay = 1e-3
target = t.tensor([1.0], requires_grad=True, dtype=t.float32)
input_idx = start_idxs[0]
output_idx = final_idx
out = optimize_input(input_idx, output_idx, target, min_steps=100, max_steps=250)

# %%
import matplotlib.pyplot as plt

plt.bar(range(out.shape[-1]), out.mean(dim=0).cpu().numpy())
plt.show()

plt.bar(range(100), model.seq[5439:](out).detach().cpu().numpy().T[0])
plt.show()


# %%
weight_decay = 0.0
output_idx = start_idxs[0] - 1
input_idx = start_idxs[1] + 1
target = out.mean(dim=0).detach()

out2 = optimize_input(
    input_idx, output_idx, target, min_steps=100, max_steps=250, inp_stddev=0.01
)

# %%

i = 1
plt.bar(range(out2.shape[-1]), out2[i].cpu().numpy())
plt.show()

plt.bar(range(48), model.seq[5437:5440](out2 * 10)[i].detach().cpu().numpy())
plt.show()

# plt.bar(range(out.shape[-1]), out[i].cpu().numpy())
# plt.show()

plt.bar(range(48), model.seq[5437:5441](out2)[i].detach().cpu().numpy())
plt.show()

# %%
plt.bar(range(out2.shape[-1]), out2[1].cpu().numpy())
plt.show()
# %%


# %%
plt.bar(range(48), model.seq[5437:5440](out2)[0].detach().cpu().numpy())
plt.show()

# %%
inp_weights = out2.detach().clone()
inp_weights.requires_grad = True

out3 = optimize_input(
    input_idx,
    final_idx,
    target=t.tensor([1.0], requires_grad=True, dtype=t.float32),
    inp_weights=inp_weights,
    min_steps=100,
    max_steps=250,
)

# %%
plt.bar(range(out3.shape[-1]), out3[i].cpu().numpy())
plt.show()

# %%
