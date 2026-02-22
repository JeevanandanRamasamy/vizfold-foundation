import torch
import os

# Let PyTorch naturally find its default cache directory (~/.cache/torch/hub)
hub_dir = torch.hub.get_dir()
checkpoint_path = os.path.join(hub_dir, "checkpoints", "esmfold_3B_v1.pt")
print(f"Loading checkpoint from {checkpoint_path}...")

# Load the weights into CPU memory
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
print("Available keys in checkpoint:", ckpt.keys())
state_dict = ckpt["model"]

# The keys that OpenFold v2.0 (VizFold) renamed
keys_to_update = [
    "trunk.structure_module.ipa.linear_q_points",
    "trunk.structure_module.ipa.linear_kv_points"
]

print("Patching keys...")
for key in keys_to_update:
    old_weight, old_bias = f"{key}.weight", f"{key}.bias"
    new_weight, new_bias = f"{key}.linear.weight", f"{key}.linear.bias"
    
    if old_weight in state_dict:
        # Move the tensor data to the new key names
        state_dict[new_weight] = state_dict.pop(old_weight)
        state_dict[new_bias] = state_dict.pop(old_bias)
        print(f"  ✓ Patched {key}")

# Save the updated checkpoint back to disk
torch.save(ckpt, checkpoint_path)
print("Success! Checkpoint is now compatible with VizFold's architecture.")
