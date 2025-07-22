from __future__ import annotations

"""ContextualResidualUNetV4 – lightweight PyTorch wrapper for inference.

This module intentionally contains *only* what is strictly required at run-time
when loading the **.safetensors** checkpoints produced by
`onnx2pytorch.ConvertModel`.  It rebuilds the computation graph from the
original ONNX file on the fly and then lets the caller load the weight tensor
state-dict.

Why rebuild from ONNX?
---------------------
• The `.safetensors` file stores **weights only** (no graph).
• The conversion utility (`onnx2pytorch`) can deterministically recreate the
  exact graph that was used when the checkpoint was exported.
• Doing it this way means we don’t need to ship hundreds of lines of frozen
  model code – we only need this tiny wrapper and an intact ONNX file sitting
  next to the checkpoint.

Usage (from SBLoadModel):
-------------------------
```
onnx_path = safetensors_path.with_suffix('.onnx')
model = ContextualResidualUNetV4Torch(onnx_path)
state = safe_load(safetensors_path, device=device)
model.load_state_dict(state, strict=False)
```
`forward()` expects the same two-input signature as the original ONNX model:
    residual = model(tile_tensor, context_tensor)

Dependencies: torch, onnx, onnx2pytorch (all small compared to TF)."""

from pathlib import Path
from typing import Union, Optional

import torch
from torch import nn

try:
    import onnx  # type: ignore
    from onnx2pytorch import ConvertModel  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "onnx and onnx2pytorch are required to rebuild the UNet graph. \n"
        "Install with:  pip install onnx onnx2pytorch\n(" + str(e) + ")"
    )


class ContextualResidualUNetV4Torch(nn.Module):
    """Reconstruct the UNet graph from an ONNX file for inference.

    Parameters
    ----------
    onnx_path : str | Path | None
        Path to the `SuperBeasts_ColorAdjustment_… .onnx` file.  If *None*, the
        class will look for a sibling file next to the calling checkpoint path
        by replacing the `.safetensors` extension with `.onnx`.
    """

    def __init__(self, onnx_path: Optional[Union[str, Path]] = None):
        super().__init__()

        # Auto-resolve ONNX path when not provided (best-effort)
        if onnx_path is None:
            # Heuristic: if the code is imported as part of SBLoadModel we can
            # inspect the call-stack to find the safetensors filename.  Fallback
            # to environment variable hook if that ever fails.
            import inspect, os

            onnx_path = None
            for frame in inspect.stack():
                if "load_file" in frame.code_context[0]:
                    safetensors_path = frame.frame.f_locals.get("model_path") or frame.frame.f_locals.get("path")
                    if safetensors_path and str(safetensors_path).lower().endswith(".safetensors"):
                        onnx_candidate = Path(safetensors_path).with_suffix(".onnx")
                        if onnx_candidate.exists():
                            onnx_path = onnx_candidate
                            break
            if onnx_path is None:
                env_hint = os.getenv("SB_ONNX_FALLBACK")
                if env_hint:
                    onnx_path = Path(env_hint)

        if onnx_path is None:
            raise FileNotFoundError(
                "Unable to locate the matching ONNX graph for ContextualResidualUNetV4. "
                "Pass `onnx_path=` explicitly or set SB_ONNX_FALLBACK to the file."
            )

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        # --- build PyTorch graph from ONNX ---
        onnx_model = onnx.load(str(onnx_path))
        torch_model = ConvertModel(onnx_model, experimental=True)
        torch_model.eval()

        # Register inner model as a sub-module so parameters appear in state_dict
        self.model = torch_model

    # The converted graph expects two inputs: tile tensor and context tensor.
    # We simply delegate the forward pass.
    def forward(self, tile_input: torch.Tensor, context_input: torch.Tensor):
        result = self.model(tile_input, context_input)
        # ConvertModel may return a tuple/list; we take first element if so.
        if isinstance(result, (list, tuple)):
            return result[0]
        return result 