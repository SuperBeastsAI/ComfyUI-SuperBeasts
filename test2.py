import torch, importlib, pathlib, sys
model_dir = pathlib.Path('torch_models')
spec = importlib.util.spec_from_file_location('ctx', model_dir / 'contextual_residual_unet_v4_pt.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
model = mod.ContextualResidualUNetV4Torch()
st = model.state_dict()
print("Model tensors:", len(st))
for i, (k, v) in enumerate(st.items()):
    if i < 40:
        print(f"{i:03d}", v.shape)