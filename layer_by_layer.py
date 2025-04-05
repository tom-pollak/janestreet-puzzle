"""
Gradient works, but because we're basically dealing with dead relus we get 0 gradient back to our input.
"""
# %%

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


linear_idxs = [i for i, l in enumerate(model.seq) if isinstance(l, t.nn.Linear)][::-1]

def optimize_layer_input(layer_idx, target_output=None, maximize=True, num_steps=1000, lr=0.1):
    """
    Optimize the input to a specific layer to either:
    1. Maximize the output (if target_output is None)
    2. Match a target output

    Returns the optimized input to the layer.
    """
    layer = model.seq[layer_idx]

    # Determine input shape
    if isinstance(layer, t.nn.Linear):
        input_shape = (layer.in_features,)
    else:
        # For ReLU, we need to determine the expected shape from the next layer
        # Find the next Linear layer to determine shape
        next_linear_idx = None
        for i in range(layer_idx+1, len(model.seq)):
            if isinstance(model.seq[i], t.nn.Linear):
                next_linear_idx = i
                break

        if next_linear_idx is not None:
            input_shape = (model.seq[next_linear_idx].in_features,)
        else:
            # Default fallback
            input_shape = (55,)

    # Initialize random input with proper shape
    layer_input = t.nn.Parameter(t.randn(input_shape, requires_grad=True))

    # Setup optimizer
    optim = t.optim.Adam([layer_input], lr=lr)

    # Optimization loop
    pbar = tqdm(range(num_steps), desc=f"Optimizing input for layer {layer_idx}")
    for i in pbar:
        optim.zero_grad()

        # Forward pass through only this layer
        output = layer(layer_input)

        # Compute loss based on objective
        if target_output is not None:
            if output.shape != target_output.shape:
                print(f"Warning: Shape mismatch: output {output.shape}, target {target_output.shape}")
                # Try to find a way to make the shapes compatible for comparison
                if len(output.shape) == len(target_output.shape) and output.shape[0] != target_output.shape[0]:
                    # If dimensions are same but sizes differ, we could:
                    # 1. Truncate the larger one
                    min_size = min(output.shape[0], target_output.shape[0])
                    loss = t.nn.functional.mse_loss(output[:min_size], target_output[:min_size])
                else:
                    # If dimensions differ completely, optimize for output magnitude instead
                    print("Shapes incompatible, switching to magnitude optimization")
                    loss = -output.mean() if maximize else output.mean()
            else:
                # Shapes match, use MSE
                loss = t.nn.functional.mse_loss(output, target_output)
        else:
            # Maximize output
            loss = -output.mean() if maximize else output.mean()

        # Backward pass
        loss.backward()

        # Update
        optim.step()

        # Occasionally print status
        if i % 100 == 0 or i == num_steps - 1:
            if target_output is not None and output.shape == target_output.shape:
                pbar.set_postfix({"MSE Loss": loss.item()})
            else:
                pbar.set_postfix({"Mean Output": -loss.item() if maximize else loss.item()})

    return layer_input.detach()

# %%

# Identify pattern of linear layers and activations
layers = []
for i, layer in enumerate(model.seq):
    if isinstance(layer, t.nn.Linear) or isinstance(layer, t.nn.ReLU):
        layers.append((i, type(layer).__name__))

print("Model structure:")
for idx, layer_type in layers:
    print(f"Layer {idx}: {layer_type}")

# %%

# Start with the last linear layer
last_linear_idx = linear_idxs[0]
print(f"Optimizing input for the last linear layer (index {last_linear_idx})")

# Optimize to maximize the output of the last linear layer
optimal_input_last_linear = optimize_layer_input(
    last_linear_idx,
    target_output=None,
    maximize=True,
    num_steps=1000,
    lr=0.1
)

# Forward pass to see what the output is
optimal_output_last_linear = model.seq[last_linear_idx](optimal_input_last_linear)
print(f"Optimal input shape: {optimal_input_last_linear.shape}")
print(f"Output after optimization: {optimal_output_last_linear.mean().item()}")

# %%

# Now work backwards through each pair of (linear, relu)
# Start from the second last linear layer
results = {}
results[last_linear_idx] = {
    'optimal_input': optimal_input_last_linear,
    'optimal_output': optimal_output_last_linear
}

current_target = optimal_input_last_linear

# Function to trace forward through layers to get output shapes
def get_layer_output_shape(start_layer_idx, input_tensor):
    with t.no_grad():
        x = input_tensor
        # Forward pass through subsequent layers
        for i in range(start_layer_idx, len(model.seq)):
            layer = model.seq[i]
            x = layer(x)
            if i == start_layer_idx:
                # Return the immediate output shape of the specified layer
                return x.shape
    return None

# Iterate backwards through linear layers
for i in range(1, len(linear_idxs)):
    current_layer_idx = linear_idxs[i]
    prev_layer_idx = linear_idxs[i-1]
    print(f"\nOptimizing input for linear layer at index {current_layer_idx}")

    # First optimize the linear layer to produce the desired output
    # We need to create an appropriate shape-compatible target
    optimal_output_shape = get_layer_output_shape(current_layer_idx, t.zeros((model.seq[current_layer_idx].in_features,)))

    if optimal_output_shape is not None and optimal_output_shape != current_target.shape:
        print(f"Shape adjustment needed: {optimal_output_shape} vs {current_target.shape}")
        # We optimize for magnitude only since shapes don't match
        optimal_input_linear = optimize_layer_input(
            current_layer_idx,
            target_output=None,  # Use None to indicate maximizing
            maximize=True,
            num_steps=1000,
            lr=0.1
        )
    else:
        # Shapes are compatible, optimize to match target
        optimal_input_linear = optimize_layer_input(
            current_layer_idx,
            target_output=current_target,
            num_steps=1000,
            lr=0.1
        )

    # Find activation layer before this linear (if any)
    activation_idx = None
    for j in range(current_layer_idx-1, -1, -1):
        if isinstance(model.seq[j], t.nn.ReLU):
            activation_idx = j
            break

    # Store result for the linear layer
    results[current_layer_idx] = {
        'optimal_input': optimal_input_linear,
        'optimal_output': model.seq[current_layer_idx](optimal_input_linear)
    }

    # If there's an activation layer before, optimize it too
    if activation_idx is not None:
        print(f"Found activation at index {activation_idx} before linear layer {current_layer_idx}")

        # The target for activation is the optimal input for the linear layer
        optimal_input_activation = optimize_layer_input(
            activation_idx,
            target_output=None,  # Use None to maximize output regardless of shape
            maximize=True,
            num_steps=1000,
            lr=0.1
        )

        # Update current target for the next iteration
        current_target = optimal_input_activation

        # Store results for activation
        results[activation_idx] = {
            'optimal_input': optimal_input_activation,
            'optimal_output': model.seq[activation_idx](optimal_input_activation)
        }
    else:
        # No activation before this linear, just use the linear's input
        current_target = optimal_input_linear

# %%

# Apply our final optimized input to the first layer and see the result through the whole model
if linear_idxs[-1] == 0:  # If the first layer is linear
    first_layer_input = results[0]['optimal_input']
else:
    # Find the first layer we optimized
    first_optimized_idx = min(results.keys())
    first_layer_input = results[first_optimized_idx]['optimal_input']

# Run the whole model with this input
with t.no_grad():
    x = first_layer_input
    layer_outputs = {-1: x}  # Store initial input at index -1

    for i, layer in enumerate(model.seq):
        x = layer(x)
        layer_outputs[i] = x

    final_output = x

print(f"Final output using layer-by-layer optimized input: {final_output.mean().item()}")

# Compare with direct optimization
print("\nComparing with direct optimization:")
pbar = tqdm(range(num_steps))
direct_input = t.nn.Parameter(t.randn((55,), requires_grad=True))
direct_optimizer = t.optim.Adam([direct_input], lr=learning_rate)

for i in pbar:
    direct_optimizer.zero_grad()
    output = model(direct_input)
    loss = -output.mean()
    loss.backward()
    direct_optimizer.step()

    if i % 100 == 0 or i == num_steps - 1:
        pbar.set_postfix({"Mean Output": -loss.item()})

print(f"Final output with direct optimization: {model(direct_input).mean().item()}")

# %%

# Visualize layer-by-layer optimization results
print("\nLayer-by-layer optimization results:")
for layer_idx in sorted(results.keys()):
    layer_type = type(model.seq[layer_idx]).__name__
    input_val = results[layer_idx]['optimal_input'].mean().item()
    output_val = results[layer_idx]['optimal_output'].mean().item()
    print(f"Layer {layer_idx} ({layer_type}): Input mean = {input_val:.4f}, Output mean = {output_val:.4f}")

# %%
