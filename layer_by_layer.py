# %%

import matplotlib.pyplot as plt
import numpy as np
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

cache = {}  # used wayy later


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
def lr_scheduler(step, n_steps, warmup_steps, min_lr, max_lr):
    assert step < n_steps
    assert warmup_steps < n_steps
    assert max_lr > min_lr

    if step < warmup_steps:
        # Linear warmup from initial_lr to final_lr
        return min_lr + (max_lr - min_lr) * (step / warmup_steps)
    else:
        # Linear decay from final_lr
        decay_factor = max(0.0, (n_steps - step) / (n_steps - warmup_steps))
        return min_lr + (max_lr - min_lr) * decay_factor


def optimize_input(
    input_idx,
    output_idx,
    target,
    inp_weights=None,
    steps=1000,
    inp_stddev=1.0,
    close_pbar=True,
    diversity_weight=0.0,
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

    else:
        assert inp_weights.shape == (
            batch_size,
            inp_dim,
        ), f"wanted {(batch_size, inp_dim)}, given {inp_weights.shape}"
        inp_weights.requires_grad = True
        inp_weights.grad = None

    if target.ndim == 1:
        target = target.unsqueeze(0).expand(batch_size, -1)

    inp = t.nn.Parameter(inp_weights)
    warmup_steps = steps // 10
    optimizer = t.optim.Adam([inp], lr=min_lr, weight_decay=weight_decay, eps=1e-5)

    for i in (pbar := tqdm(range(steps))):
        current_lr = lr_scheduler(i, steps, warmup_steps, min_lr, max_lr)
        optimizer.param_groups[0]["lr"] = current_lr

        optimizer.zero_grad()
        sub_model = model.seq[input_idx : output_idx + 1]
        acts = sub_model(inp)
        # mse_loss = t.nn.functional.mse_loss(acts, target)
        euclidean_loss = t.nn.functional.pairwise_distance(acts, target).mean()
        euclidean_loss = euclidean_loss * euclidean_weight

        # Calculate the cosine embedding loss using the functional interface
        cosine_loss = t.nn.functional.cosine_embedding_loss(
            acts,
            target,
            target=t.ones(acts.shape[0], device=acts.device),
            margin=0.0,
            reduction="mean",
        )
        cosine_loss = cosine_loss * cosine_weight

        dist_loss = euclidean_loss + cosine_loss

        normalized_inp = t.nn.functional.normalize(inp, p=2, dim=-1)
        cosine_sim = normalized_inp @ normalized_inp.T
        mask = t.tril(t.ones_like(cosine_sim), diagonal=-1)
        # mean cosine similarity between different inputs
        cossim_loss = (cosine_sim * mask).sum() / (mask.sum() + 1e-8)

        loss = dist_loss + diversity_weight * cossim_loss

        loss.backward()
        grad_norm = t.nn.utils.clip_grad_norm_(inp, max_norm=max_grad_norm)
        optimizer.step()

        if i % 250 == 0:
            pbar.set_postfix(
                loss=loss.item(),
                euclidean_loss=euclidean_loss.item(),
                cosine_loss=cosine_loss.item(),
                cossim_loss=cossim_loss.item(),
                grad_norm=grad_norm.item(),
            )

        if loss < break_loss and i > steps // 10:
            break

    if close_pbar:
        pbar.close()
    return inp.detach()


# %%

# # need weight decay for this one
# weight_decay = 1e-3
# target = t.tensor([1.0], requires_grad=True, dtype=t.float32)
# input_idx = start_idxs[0]
# output_idx = final_idx
# out = optimize_input(input_idx, output_idx, target, steps=250)

# # %%

# plt.bar(range(out.shape[-1]), out.mean(dim=0).cpu().numpy())
# plt.show()

# plt.bar(range(100), model.seq[5439:](out).detach().cpu().numpy().T[0])
# plt.show()


# # %%
# weight_decay = 1e-2  # 0.0
# output_idx = start_idxs[0] - 1
# input_idx = start_idxs[1]  # + 1
# target = out.mean(dim=0).detach()

# out2 = optimize_input(input_idx, output_idx, target, steps=10000, inp_stddev=1.0)
# # %%

# i = 1
# plt.bar(range(out2.shape[-1]), out2[i].cpu().numpy())
# plt.show()

# plt.bar(range(48), model.seq[5437:5440](out2 * 10)[i].detach().cpu().numpy())
# plt.show()

# plt.bar(range(48), model.seq[5437:5441](out2)[i].detach().cpu().numpy())
# plt.show()

# plt.bar(range(48), model.seq[5437:5440](out2)[0].detach().cpu().numpy())
# plt.show()

# # %%
# print(model.seq[5437:5441](out2[0]))


# # %%
# min_lr = 1e-6
# max_lr = 5e-6
# weight_decay = 0.0
# max_grad_norm = 0.1


# inp_weights = out2.detach().clone()
# inp_weights.requires_grad = True

# out3 = optimize_input(
#     input_idx,
#     final_idx,
#     target=t.tensor([1.0], requires_grad=True, dtype=t.float32),
#     inp_weights=inp_weights,
#     steps=5000,
# )

# plt.bar(range(out3.shape[-1]), out3[i].cpu().numpy())
# plt.show()

# print(model.seq[5437:5441](out3[0]))

# %%


# ████████████████████████████████  Do The Thing  ████████████████████████████████


batch_size = 100
min_lr = 1e-3
max_lr = 5e-3
break_loss = 1e-8
weight_decay = 1e-3
max_grad_norm = 1.0
diversity_weight = 0.0  # 1e-10
euclidean_weight = 1.0
cosine_weight = 0.0

# %%

# I think for the first layer, we need a batch to mean out all the different possible ways of getting a 1.0
# we can actually remove batch_size after this

target = t.tensor([1.0], requires_grad=True, dtype=t.float32)
input_idx = start_idxs[0]
output_idx = final_idx
final_output = optimize_input(
    input_idx, output_idx, target, steps=10000, diversity_weight=1e-6
)

mean_final_output = final_output.mean(dim=0).detach()

# %%
plt.bar(range(mean_final_output.shape[-1]), mean_final_output.cpu().numpy())
plt.title(
    f"Final layer optimization {model.seq[5439:final_idx+1](mean_final_output).detach().cpu().numpy().item()}",
)
plt.show()


# %%


def plot_layer(inp, input_idx, output_idx, final_idx, title):
    assert inp.ndim == 1
    inp = inp.unsqueeze(0)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    to_final_layer = model.seq[input_idx : final_idx + 1]
    final_layer_outp = to_final_layer(inp).detach().item()

    fig.suptitle(f"{title} -- {final_layer_outp:.3f}")

    ax1.set_title("Features")
    ax1.bar(range(inp.shape[-1]), inp.cpu().numpy()[0])

    next_layer = model.seq[input_idx : output_idx + 1]
    next_layer_outp = next_layer(inp)[0].detach().cpu().numpy()

    print(next_layer_outp.shape)
    ax2.set_title("Map to next layer")
    ax2.bar(range(next_layer_outp.shape[-1]), next_layer_outp)

    plt.show()


def optimize_layer(prev_output, i, steps):
    global min_lr, max_lr

    min_lr = 1e-3
    max_lr = 5e-3
    output_idx = (
        start_idxs[i - 1] - 1
    )  # off the relu because we don't want to kill signal
    input_idx = start_idxs[i] # + 1 ### off the relu again # on the relu

    # this gets us to an ok point in input space to start our optimization, so we actually get a gradient.
    temp_input = optimize_input(
        input_idx,
        output_idx,
        prev_output,
        steps=steps,
        diversity_weight=diversity_weight,
    )

    temp_model_outp = model.seq[input_idx : final_idx + 1](temp_input)
    best_temp_output_idx = temp_model_outp.argmax()

    plot_layer(
        temp_input[best_temp_output_idx],
        input_idx,
        output_idx,
        final_idx,
        f"{i} (temp)",
    )

    inp_weights = temp_input.detach().clone()
    inp_weights.requires_grad = True

    min_lr = 1e-6
    max_lr = 5e-6

    final_input = optimize_input(
        input_idx,
        final_idx,
        target=t.tensor([1.0], requires_grad=True, dtype=t.float32),
        inp_weights=inp_weights,
        steps=steps,
        diversity_weight=0.0,
    )

    model_outp = model.seq[input_idx : final_idx + 1](final_input)
    best_output_idx = model_outp.argmax()
    best_output = model_outp[best_output_idx].item()

    plot_layer(
        final_input[best_output_idx],
        input_idx,
        final_idx,
        final_idx,
        title=f"{i} (final)",
    )

    terminate = best_output < 0

    return temp_input, final_input, terminate


# %%


# prev_output = final_output
prev_output = mean_final_output
# prev_output = a
# batch_size = a.shape[0]

# batch_size = prev_output.shape[0]
batch_size = 512

def resume_from_layer(i):
    print(f"Resuming from layer {i}")
    global prev_output
    temp_input, final_input = cache[i]
    prev_output = final_input
    # prev_output = temp_input


start_layer = 1
if (start_layer - 1) in cache:
    resume_from_layer(start_layer - 1)
else:
    assert start_layer == 1, f"{start_layer} is not in cache"


for i in tqdm(range(start_layer, len(start_idxs))):
    p = t.nn.functional.relu(prev_output)
    temp_input, final_input, terminate = optimize_layer(
        p, i, steps=10000
    )
    cache[i] = (temp_input, final_input)
    prev_output = final_input
    # prev_output = temp_input
    # could also try to mean the final_input together to get the best?
    # no exploration this way though

    if terminate:
        print(f"Terminated at layer {i}!")
        break

# %%

batch_size = 512
end = model.seq[start_idxs[2] : final_idx + 1]

n_tries = 10_000_000
inps = []
acts = []
for _ in tqdm(range(n_tries // batch_size)):
    with t.no_grad():
        b_inps = t.randn(batch_size, 320)
        b_acts = end(b_inps)
        inps.append(b_inps.cpu())
        acts.append(b_acts.cpu())

inps = t.cat(inps, dim=0)
acts = t.cat(acts, dim=0)
inps.shape, acts.shape

# %%

values, indices = acts.squeeze().topk(512)
best_acts = acts[indices]

best_inps = inps[indices]
values

# %%

inp_weights = best_inps.detach().clone()
inp_weights = inp_weights.to(device)
inp_weights.requires_grad = True

min_lr = 1e-6
max_lr = 5e-6

input_idx = start_idxs[2]

final_input = optimize_input(
    input_idx,
    final_idx,
    target=t.tensor([1.0], requires_grad=True, dtype=t.float32),
    inp_weights=inp_weights,
    steps=10_000,
    diversity_weight=0.0,
)

model_outp = model.seq[input_idx : final_idx + 1](final_input)
best_output_idx = model_outp.argmax()
best_output = model_outp[best_output_idx].item()

plot_layer(
    final_input[best_output_idx],
    input_idx,
    final_idx,
    final_idx,
    title=f"adf",
)


# %%


# prev_output_1 = cache[1][1]
# a = model.seq[start_idxs[1] : final_idx + 1](prev_output_1)
# a
# # %%
#
# x = t.nn.functional.normalize(prev_output_1, p=2, dim=-1).cpu().numpy()
# plt.imshow(x @ x.T)
#
# # %%
# third_last = model.seq[5436]
#
# plt.imshow(third_last.weight.data.cpu().numpy())
# plt.show()
# plt.bar(range(third_last.bias.data.shape[-1]), third_last.bias.data.cpu().numpy())
# plt.show()
#
# # %%
#
# a = t.zeros(320, dtype=t.float32, device=device)
# # a[t.arange(25)] = 1.
# a[t.from_numpy(idxs)] = 1.0
# third_last = third_last.to(device)
# out = third_last(a).cpu().detach().numpy()
# plt.bar(range(out.shape[-1]), out)
# plt.show()
#
# # %%
#
#
#
#
# idxs = np.argwhere(third_last.weight.data.cpu().numpy() > 0)
# third_last.weight.data.cpu().numpy()[idxs[:, 0], idxs[:, 1]]
# # %%
#
# plt.imshow(last.weight.data.cpu().numpy())
# plt.show()
# plt.bar(range(last.bias.data.shape[-1]), last.bias.data.cpu().numpy())
# plt.show()
#
# # %%
#
#
# # %%
# last = model.seq[5440]
# w = last.weight.data
# idxs = (w > 0).argwhere()[:, 1]
# a = t.zeros(idxs.shape[0], 48).to(device)
# a[t.arange(idxs.shape[0]), idxs] = 16.0
# plt.imshow(a.cpu().numpy())
# plt.show()
#
# # %%
#
# model.seq[0].weight.data
#
#
# # %%
#
