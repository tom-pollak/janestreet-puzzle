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
weight_decay = 0.0 # 1e-3
batch_size = 100
# %%


def lr_scheduler(step, min_steps, max_steps, initial_lr, warmup_steps, final_lr):
    if step < warmup_steps:
        return initial_lr + (1e-1 - initial_lr) * (step / warmup_steps)
    elif step < min_steps:
        return 1e-1 - (1e-1 - final_lr) * (
            (step - warmup_steps) / (min_steps - warmup_steps)
        )
    else:
        return final_lr


def optimize_input(
    input_idx, output_idx, target, inp_weights=None, min_steps=100, max_steps=1000
):
    input_layer = model.seq[input_idx]
    if not hasattr(input_layer, "out_features"):  # isinstance(input_layer, t.nn.ReLU):
        inp_dim = model.seq[input_idx - 1].out_features
    else:
        inp_dim = input_layer.in_features

    if inp_weights is None:
        inp_weights = t.randn(batch_size, inp_dim, requires_grad=True, dtype=t.float32) # * 0.01

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

    if target.ndim == 1:
        target = target.unsqueeze(0).expand(batch_size, -1)

    inp = t.nn.Parameter(inp_weights)
    warmup_steps = min_steps // 10
    optimizer = t.optim.Adam([inp], lr=min_lr, weight_decay=weight_decay)

    pbar = tqdm(range(max_steps))
    for i in pbar:
        current_lr = lr_scheduler(i, min_steps, max_steps, min_lr, warmup_steps, max_lr)
        optimizer.param_groups[0]["lr"] = current_lr

        optimizer.zero_grad()
        with sight.trace(rand_toks) as tracer:
            sight.seq[input_idx].input = inp
            acts = sight.seq[output_idx].output.save()
            # diff_loss = t.nn.functional.mse_loss(acts, target)
            if target.shape == (batch_size, 1):
                diff_loss = t.nn.functional.mse_loss(acts, target)
            else:
                diff_loss = t.nn.functional.cosine_embedding_loss(
                    acts, target, target=t.ones(batch_size)
                )
            diff_loss.save()

        # print(acts.shape, target.shape)
        zinp = t.nn.functional.normalize(inp, p=2, dim=-1)
        bottom_tri = t.tril(t.ones_like(zinp @ zinp.T), diagonal=-1)
        cossim = (zinp @ zinp.T) * bottom_tri
        cossim_loss = t.nn.functional.mse_loss(cossim, bottom_tri)

        loss = diff_loss + cossim_loss * cos_alpha
        loss.backward()
        optimizer.step()
        grad_norm = inp.grad.norm()

        pbar.set_postfix(
            loss=loss.item(),
            diff_loss=diff_loss.item(),
            cossim_loss=cossim_loss.item(),
            grad_norm=grad_norm.item(),
        )

        if diff_loss < eps_mse and cossim_loss < eps_cossim and i > min_steps:
            break

    pbar.close()
    return inp.detach()


# %%

target = t.tensor([1.0], requires_grad=True, dtype=t.float32)
input_idx = start_idxs[0]
output_idx = final_idx
out = optimize_input(input_idx, output_idx, target, min_steps=100, max_steps=250)

# %%
import matplotlib.pyplot as plt

plt.bar(range(out.shape[-1]), out.mean(dim=0).cpu().numpy())
plt.show()

# %%
model.seq[5439:](out)


# %%
output_idx = start_idxs[0] - 1
input_idx = start_idxs[1] + 1
target = out.mean(dim=0).detach()

out2 = optimize_input(input_idx, output_idx, target, min_steps=100, max_steps=250)

# %%

i = 1
plt.bar(range(out2.shape[-1]), out2[i].cpu().numpy())
plt.show()

plt.bar(range(48), model.seq[5437:5440](out2*10)[i].detach().cpu().numpy())
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
plt.bar(range(48), model.seq[5437:5439](t.randn(1, 192))[0].detach().cpu().numpy())
plt.show()
