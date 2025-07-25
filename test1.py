from safetensors.torch import load_file
state = load_file('models/SuperBeasts_ColorAdjustment_512px_V1.safetensors')
print("Checkpoint tensors:", len(state))
for i, (k, v) in enumerate(state.items()):
    if i < 40:  # adjust if you want more
        print(f"{i:03d}", v.shape)