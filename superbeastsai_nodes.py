import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageEnhance, ImageCms
from PIL.PngImagePlugin import PngInfo
import torch
import torch.nn.functional as F
import json
import random
import os
import re
import urllib.request, pathlib, shutil, tempfile
from typing import Dict, Any, List
import importlib.util
import sys


# Use importlib to probe for the optional global `config` module without triggering static
# import-time linter errors when it does not exist in the current environment.
_cfg_spec = importlib.util.find_spec("config")
if _cfg_spec is not None:
    _cfg = importlib.util.module_from_spec(_cfg_spec)  # type: ignore[var-annotated]
    _cfg_spec.loader.exec_module(_cfg)  # type: ignore[arg-type]
    CONTEXT_THUMBNAIL_SIZE = getattr(_cfg, "CONTEXT_THUMBNAIL_SIZE", (64, 64))
else:
    CONTEXT_THUMBNAIL_SIZE = (64, 64)

# ---------------------------------------------------------------------------
# Remote model download helper
# ---------------------------------------------------------------------------

# Base URL for hosting – we append <family>/<version>/<filename>
_REMOTE_MODEL_BASE_URL = "https://raw.githubusercontent.com/SuperBeastsAI/SuperBeastsAI-Models/main/"
# License info for downloaded model weights
_SPCA_LICENSE_ID = "SPCA-Community-NoSaaS"
_SPCA_LICENSE_URL = (
    "https://github.com/SuperBeastsAI/SuperBeastsAI-Models/"
    "blob/main/SuperPopColorAdjustment/LICENSE.txt"
)
# File written into models dir after first download so we don't spam the console
_SPCA_LICENSE_STUB_FILENAME = "SPCA_LICENSE_STUB.txt"
# Env var: set to "1" (or any truthy string) to suppress the banner (CI / headless automation)
_SUPERBEASTS_SILENCE_LICENSE_ENV = "SUPERBEASTS_SILENCE_LICENSE"


# Registry of published model families → version → filename
_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "SuperPopColorAdjustment": {
        "v1.0": "SuperBeasts_ColorAdjustment_512px_V1.onnx",
    },
    # Future models / versions can be added here
}


def _latest_version(family: str) -> str:
    versions = list(_MODEL_REGISTRY.get(family, {}).keys())
    if not versions:
        raise ValueError(f"Unknown model family: {family}")
    # assume semantic ‘vX.Y’ strings – sort lexically
    return sorted(versions)[-1]


def _download_remote_model(rel_path: str, dest_path: str):
    """Download file at *rel_path* (relative to base URL) to *dest_path*.
    Emits a one-time license banner and writes a stub file so we don't repeat warnings.
    """
    url = _REMOTE_MODEL_BASE_URL + rel_path
    print(f"[SuperBeasts] Downloading model weights from {url} …")

    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, rel_path.split('/')[-1] + ".part")

    try:
        # --- download file ---
        with urllib.request.urlopen(url) as r, open(tmp_file, "wb") as f:
            shutil.copyfileobj(r, f)

        # --- move into place ---
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(tmp_file, dest_path)
        print(f"[SuperBeasts] Saved weights to {dest_path}")

        # --- after successful download, print license notice (once per repo install) ---
        _maybe_print_spca_license_notice(os.path.dirname(dest_path))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Import helper to add extra channels same way training did
try:
    from .extras.inference.inference_utils import add_extra_color_channels_np as _add_extra_color_channels_np  # type: ignore
except Exception:
    _add_extra_color_channels_np = None  # will define below

# ------------------------------------------------------------------
# Ensure we always add the extra YCbCr (+3) and Saturation (+1) channels so
# the input tensor matches the 7-channel format expected by the SuperBeasts
# models. The helper below mirrors the logic from inference_utils but forces
# the extras on regardless of global config flags.
# ------------------------------------------------------------------

try:
    import cv2  # Optional – used for fast RGB→HSV conversion
    _HAS_CV2 = True
except ImportError:  # linter environments might not have it
    _HAS_CV2 = False

def _sb_add_extra_channels(arr: np.ndarray) -> np.ndarray:
    """Return RGB array with YCbCr and saturation channels appended (total 7)."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Already has extra channels or malformed – return unchanged
        return arr

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # YCbCr as in training
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    ycbcr = np.stack([y, cb, cr], axis=-1).astype(arr.dtype)

    # Saturation channel – use OpenCV if available, otherwise approximate
    if _HAS_CV2:
        hsv = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
        sat = (hsv[:, :, 1] / 255.0).astype(arr.dtype)
    else:
        # Simple saturation approximation without HSV conversion
        max_rgb = arr.max(axis=2)
        min_rgb = arr.min(axis=2)
        sat = np.where(max_rgb == 0, 0.0, (max_rgb - min_rgb) / max_rgb).astype(arr.dtype)
    sat = sat[:, :, np.newaxis]

    return np.concatenate([arr, ycbcr, sat], axis=-1).astype(arr.dtype)

# If we imported the original util and it already produces ≥7 channels we keep it.
if _add_extra_color_channels_np is None:
    _add_extra_color_channels_np = _sb_add_extra_channels
else:
    # Wrap the existing util so we guarantee at least 7 channels.
    _orig_fn = _add_extra_color_channels_np

    def _add_extra_color_channels_np(arr):  # type: ignore[override]
        out = _orig_fn(arr)
        if out.shape[2] < 7:
            out = _sb_add_extra_channels(out)
        return out

# ---------------------------------------------
# SB MODEL LOADER NODE
# ---------------------------------------------

def _maybe_print_spca_license_notice(models_dir: str):
    """
    Print the SPCA license banner once per install.
    We create a stub file in the models_dir so we don't repeat on future runs.
    Suppress by setting env var SUPERBEASTS_SILENCE_LICENSE=1.
    """
    if os.environ.get(_SUPERBEASTS_SILENCE_LICENSE_ENV, "").strip():
        return

    stub_path = os.path.join(models_dir, _SPCA_LICENSE_STUB_FILENAME)
    if os.path.exists(stub_path):
        # Already shown
        return

    banner = (
        "\n"
        "============================================================\n"
        " Super Pop Color Adjustment – Model License Notice\n"
        "------------------------------------------------------------\n"
        f" Weights licensed under {_SPCA_LICENSE_ID}.\n"
        " Personal / client / local use OK.\n"
        " Public SaaS/API redistribution requires a license.\n"
        f" Full terms: {_SPCA_LICENSE_URL}\n"
        "============================================================\n"
    )
    print(banner, file=sys.stderr)

    # write stub file so we don't print again
    try:
        with open(stub_path, "w", encoding="utf-8") as f:
            f.write(
                "Super Pop Color Adjustment weights downloaded.\n"
                "No public SaaS/API redistribution without license. Contact: Via DM on Instagram @SuperBeasts.AI"
                f"License: {_SPCA_LICENSE_ID}\n"
                f"See: {_SPCA_LICENSE_URL}\n"
            )
    except Exception as e:
        # non-fatal; just warn
        print(f"[SuperBeasts] WARNING: could not write license stub: {e}", file=sys.stderr)


def _discover_sb_models(models_dir: str):
    """Return a list of available .keras / .h5 model filenames (relative, not absolute)."""
    if not os.path.isdir(models_dir):
        return []
    names = []
    for file in os.listdir(models_dir):
        if file.lower().endswith(('.keras', '.h5', '.safetensors', '.onnx')):
            names.append(file)
    # Sort for stable dropdown order
    return sorted(names)


class SBLoadModel:
    """Load a SuperBeasts colour-adjustment model (downloads if missing)./n
    
    Select a model from the dropdown. Use Family/Version entries (e.g., SuperPopColorAdjustment/latest)
    to auto-download from the official model registry when not found locally./n
    
    ⚠ License: Downloaded weights are licensed under SPCA-Community-NoSaaS.
    Local / personal / commercial use OK. SaaS/API redistribution using this model requires a license.
    See: https://github.com/SuperBeastsAI/SuperBeastsAI-Models/tree/main/SuperPopColorAdjustment
    """
    # Shown in ComfyUI when the user clicks the "?" icon on the node title-bar.
    DESCRIPTION = __doc__

    _model_cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def INPUT_TYPES(cls):
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        available = _discover_sb_models(models_dir)

        dropdown_set = set(available)
        # Add registry family/version keys so saved workflows remain valid
        for fam, vers_dict in _MODEL_REGISTRY.items():
            dropdown_set.add(f"{fam}/latest")
            for ver in vers_dict.keys():
                dropdown_set.add(f"{fam}/{ver}")

        dropdown = sorted(dropdown_set)
        return {
            "required": {
                "model_key": (dropdown, ),
                "device": (["AUTO", "CPU", "GPU"], {"default": "AUTO"}),
            }
        }

    RETURN_TYPES = ("SBMODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = "load"
    CATEGORY = "SuperBeastsAI/Model"

    def _parse_patch_size(self, filename: str, default: int = 512):
        m = re.search(r"(\d+)px", filename)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return default

    def load(self, model_key: str, device: str = "AUTO"):
        """`model_key` can be either a raw filename (legacy) *or* the form
        "Family/Version" (e.g. "SuperPopColorAdjustment/v1.0" or "SuperPopColorAdjustment/latest")."""

        # ------------------------------------------------------------------
        # Resolve model filename, family, version
        # ------------------------------------------------------------------
        if "/" in model_key:
            family, version = model_key.split("/", 1)
            if version.lower() == "latest":
                version = _latest_version(family)
            try:
                filename = _MODEL_REGISTRY[family][version]
            except KeyError:
                raise ValueError(f"Unknown model/version combination: {model_key}")
        else:
            # Legacy direct filename path (assume SuperPop)…
            family = "SuperPopColorAdjustment"
            version = _latest_version(family)
            filename = model_key

        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, filename)

        if not os.path.isfile(model_path):
            rel_path = f"{family}/{version}/{filename}"
            try:
                _download_remote_model(rel_path, model_path)
            except Exception as e:
                raise FileNotFoundError(
                    f"Model file not found locally and automatic download failed.\n"
                    f"Attempted path: {rel_path}\n{e}"
                )

        # Use cache if possible
        if model_path in self._model_cache:
            return (self._model_cache[model_path], )

        global tf, keras  # ensure global vars updated
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "onnxruntime is required for SuperBeasts models. Install with:\n"
                " pip install onnxruntime-gpu  # (or onnxruntime for CPU)\n\n" + str(e)
            )

        # Optionally set device context
        dev_ctx = None
        if device == "GPU":
            dev_ctx = "/GPU:0"
        elif device == "CPU":
            dev_ctx = "/CPU:0"

        # Determine patch size from filename
        patch_size = self._parse_patch_size(filename)

        if model_path.lower().endswith('.safetensors'):
            import torch
            from safetensors.torch import load_file as safe_load
            import importlib.util, sys, pathlib
            model_dir = pathlib.Path(__file__).parent / 'torch_models'
            sys.path.append(str(model_dir))
            spec = importlib.util.spec_from_file_location('contextual_residual_unet_v4', model_dir / 'contextual_residual_unet_v4.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            ContextualResidualUNetV4Torch = module.ContextualResidualUNetV4Torch

            device_str = 'cuda' if device == 'GPU' and torch.cuda.is_available() else 'cpu'
            model_pt = ContextualResidualUNetV4Torch()
            state = safe_load(model_path, device=device_str)
            model_pt.load_state_dict(state, strict=False)
            model_pt.eval().to(device_str)

            model_info = {
                "model": model_pt,
                "device": device_str,
                "patch_size": patch_size,
                "type": "torch",
            }
        else:
            # Load ONNX with onnxruntime
            providers: List[str]
            avail_prov = ort.get_available_providers()
            print(f"[SuperBeasts] ONNX Runtime providers available: {avail_prov}")
            cuda_available = 'CUDAExecutionProvider' in avail_prov
            dml_available = 'DmlExecutionProvider' in avail_prov

            chosen_provider = 'CPUExecutionProvider'

            if device == "GPU":
                if cuda_available:
                    chosen_provider = 'CUDAExecutionProvider'
                elif dml_available:
                    chosen_provider = 'DmlExecutionProvider'
            elif device == "AUTO":
                if cuda_available:
                    chosen_provider = 'CUDAExecutionProvider'
                elif dml_available:
                    chosen_provider = 'DmlExecutionProvider'

            if chosen_provider != 'CPUExecutionProvider':
                providers = [chosen_provider, 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            print(f"[SuperBeasts] Using ORT providers: {providers}")

            # Create session
            session_opts = ort.SessionOptions()
            session_opts.log_severity_level = 3  # suppress info logs
            session = ort.InferenceSession(model_path, session_options=session_opts, providers=providers)

            input_names = [i.name for i in session.get_inputs()]

            model_info = {
                "session": session,
                "input_names": input_names,
                "patch_size": patch_size,
                "path": model_path,
                "type": "onnx",
            }

        # Cache and return
        self._model_cache[model_path] = model_info
        return (model_info, )


sRGB_profile = ImageCms.createProfile("sRGB")
Lab_profile = ImageCms.createProfile("LAB")

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def adjust_shadows(luminance_array, shadow_intensity, hdr_intensity):
    # Darken shadows more as shadow_intensity increases, scaled by hdr_intensity
    return np.clip(luminance_array - luminance_array * shadow_intensity * hdr_intensity * 0.5, 0, 255)

def adjust_highlights(luminance_array, highlight_intensity, hdr_intensity):
    # Brighten highlights more as highlight_intensity increases, scaled by hdr_intensity
    return np.clip(luminance_array + (255 - luminance_array) * highlight_intensity * hdr_intensity * 0.5, 0, 255)

def apply_adjustment(base, factor, intensity_scale):
    """Apply positive adjustment scaled by intensity."""
    # Ensure the adjustment increases values within [0, 1] range, scaling by intensity
    adjustment = base + (base * factor * intensity_scale)
    # Ensure adjustment stays within bounds
    return np.clip(adjustment, 0, 1)


def multiply_blend(base, blend):
    """Multiply blend mode."""
    return np.clip(base * blend, 0, 255)

def overlay_blend(base, blend):
    """Overlay blend mode."""
    # Normalize base and blend to [0, 1] for blending calculation
    base = base / 255.0
    blend = blend / 255.0
    return np.where(base < 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend)) * 255

def adjust_shadows_non_linear(luminance, shadow_intensity, max_shadow_adjustment=1.5):
    lum_array = np.array(luminance, dtype=np.float32) / 255.0  # Normalize
    # Apply a non-linear darkening effect based on shadow_intensity
    shadows = lum_array ** (1 / (1 + shadow_intensity * max_shadow_adjustment))
    return np.clip(shadows * 255, 0, 255).astype(np.uint8)  # Re-scale to [0, 255]

def adjust_highlights_non_linear(luminance, highlight_intensity, max_highlight_adjustment=1.5):
    lum_array = np.array(luminance, dtype=np.float32) / 255.0  # Normalize
    # Brighten highlights more aggressively based on highlight_intensity
    highlights = 1 - (1 - lum_array) ** (1 + highlight_intensity * max_highlight_adjustment)
    return np.clip(highlights * 255, 0, 255).astype(np.uint8)  # Re-scale to [0, 255]

def merge_adjustments_with_blend_modes(luminance, shadows, highlights, hdr_intensity, shadow_intensity, highlight_intensity):
    # Ensure the data is in the correct format for processing
    base = np.array(luminance, dtype=np.float32)
    
    # Scale the adjustments based on hdr_intensity
    scaled_shadow_intensity = shadow_intensity ** 2 * hdr_intensity
    scaled_highlight_intensity = highlight_intensity ** 2 * hdr_intensity
    
    # Create luminance-based masks for shadows and highlights
    shadow_mask = np.clip((1 - (base / 255)) ** 2, 0, 1)
    highlight_mask = np.clip((base / 255) ** 2, 0, 1)
    
    # Apply the adjustments using the masks
    adjusted_shadows = np.clip(base * (1 - shadow_mask * scaled_shadow_intensity), 0, 255)
    adjusted_highlights = np.clip(base + (255 - base) * highlight_mask * scaled_highlight_intensity, 0, 255)
    
    # Combine the adjusted shadows and highlights
    adjusted_luminance = np.clip(adjusted_shadows + adjusted_highlights - base, 0, 255)
    
    # Blend the adjusted luminance with the original luminance based on hdr_intensity
    final_luminance = np.clip(base * (1 - hdr_intensity) + adjusted_luminance * hdr_intensity, 0, 255).astype(np.uint8)

    return Image.fromarray(final_luminance)

def apply_gamma_correction(lum_array, gamma):
    """
    Apply gamma correction to the luminance array.
    :param lum_array: Luminance channel as a NumPy array.
    :param gamma: Gamma value for correction.
    """
    if gamma == 0:
        return np.clip(lum_array, 0, 255).astype(np.uint8)

    epsilon = 1e-7  # Small value to avoid dividing by zero
    gamma_corrected = 1 / (1.1 - gamma)
    adjusted = 255 * ((lum_array / 255) ** gamma_corrected)
    return np.clip(adjusted, 0, 255).astype(np.uint8)
    
# create a wrapper function that can apply a function to multiple images in a batch while passing all other arguments to the function
def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        images = []
        for img in image:
            images.append(func(self, img, *args, **kwargs))
        batch_tensor = torch.cat(images, dim=0)
        return (batch_tensor, )
    return wrapper

class HDREffects:
    DESCRIPTION = "Apply an HDR-style tone-mapping effect with separate control over shadows, highlights, gamma, contrast and colour boost. Accepts a batch of images and returns the processed batch."
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE', {'default': None}),
                             'hdr_intensity': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 5.0, 'step': 0.01}),
                             'shadow_intensity': ('FLOAT', {'default': 0.25, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                             'highlight_intensity': ('FLOAT', {'default': 0.75, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                             'gamma_intensity': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                             'contrast': ('FLOAT', {'default': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                             'enhance_color': ('FLOAT', {'default': 0.25, 'min': 0.0, 'max': 1.0, 'step': 0.01})
                             }}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('result_img',)
    FUNCTION = 'apply_hdr2'
    CATEGORY = 'SuperBeastsAI/Image'
    
    @apply_to_batch
    def apply_hdr2(self, image, hdr_intensity=0.5, shadow_intensity=0.25, highlight_intensity=0.75, gamma_intensity=0.25, contrast=0.1, enhance_color=0.25):
        # Load the image
        img = tensor2pil(image)
        
        # Step 1: Convert RGB to LAB for better color preservation
        img_lab = ImageCms.profileToProfile(img, sRGB_profile, Lab_profile, outputMode='LAB')

        # Extract L, A, and B channels
        luminance, a, b = img_lab.split()
        
        # Convert luminance to a NumPy array for processing
        lum_array = np.array(luminance, dtype=np.float32)

        # Preparing adjustment layers (shadows, midtones, highlights)
        # This example assumes you have methods to extract or calculate these adjustments
        shadows_adjusted = adjust_shadows_non_linear(luminance, shadow_intensity)
        highlights_adjusted = adjust_highlights_non_linear(luminance, highlight_intensity)


        merged_adjustments = merge_adjustments_with_blend_modes(lum_array, shadows_adjusted, highlights_adjusted, hdr_intensity, shadow_intensity, highlight_intensity)

        # Apply gamma correction with a base_gamma value (define based on desired effect)
        gamma_corrected = apply_gamma_correction(np.array(merged_adjustments), gamma_intensity)
        gamma_corrected = Image.fromarray(gamma_corrected).resize(a.size)


        # Merge L channel back with original A and B channels
        adjusted_lab = Image.merge('LAB', (gamma_corrected, a, b))

        # Step 3: Convert LAB back to RGB
        img_adjusted = ImageCms.profileToProfile(adjusted_lab, Lab_profile, sRGB_profile, outputMode='RGB')
        
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img_adjusted)
        contrast_adjusted = enhancer.enhance(1 + contrast)

        
        # Enhance color saturation
        enhancer = ImageEnhance.Color(contrast_adjusted)
        color_adjusted = enhancer.enhance(1 + enhance_color * 0.2)
         
        return pil2tensor(color_adjusted)

class MakeResizedMaskBatch:
    DESCRIPTION = "Combine up to 12 individual masks/batches into one mask batch, automatically sizing/cropping each mask to the requested width/height. Useful for building consistent video masks."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "step": 1}),
            },
            "optional": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "mask3": ("MASK",),
                "mask4": ("MASK",),
                "mask5": ("MASK",),
                "mask6": ("MASK",),
                "mask7": ("MASK",),
                "mask8": ("MASK",),
                "mask9": ("MASK",),
                "mask10": ("MASK",),
                "mask11": ("MASK",),
                "mask12": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "append"
    CATEGORY = "SuperBeastsAI/Masks"

    def append(self, width, height, mask1=None, mask2=None, mask3=None, mask4=None, mask5=None, mask6=None,
               mask7=None, mask8=None, mask9=None, mask10=None, mask11=None, mask12=None):
        masks = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10, mask11, mask12]
        valid_masks = [mask for mask in masks if mask is not None]

        if not valid_masks:
            raise ValueError("At least one input mask must be provided.")

        cropped_masks = []
        for mask in valid_masks:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif mask.ndim == 3:
                mask = mask.unsqueeze(0)  # Add batch dimension
            elif mask.ndim != 4 or (mask.ndim == 4 and mask.shape[1] != 1):
                raise ValueError(f"Invalid mask shape: {mask.shape}. Expected (N, 1, H, W) or (1, H, W) or (H, W).")

            # Scale the mask to match the desired width while maintaining the aspect ratio
            scale_factor = width / mask.shape[-1]
            scaled_height = int(mask.shape[-2] * scale_factor)
            scaled_mask = F.interpolate(mask, size=(scaled_height, width), mode='bilinear', align_corners=False)

            # Perform center cropping
            if scaled_height < height:
                # Pad the top and bottom of the mask
                pad_top = (height - scaled_height) // 2
                pad_bottom = height - scaled_height - pad_top
                cropped_mask = F.pad(scaled_mask, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
            else:
                # Crop the center of the mask
                crop_top = (scaled_height - height) // 2
                crop_bottom = crop_top + height
                cropped_mask = scaled_mask[:, :, crop_top:crop_bottom, :]

            cropped_masks.append(cropped_mask)

        # Concatenate the cropped masks along the batch dimension
        result = torch.cat(cropped_masks, dim=0)

        return (result,)



def adjust_brightness(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(brightness_factor)
    return adjusted_image

def calculate_brightness_factor(target_brightness, current_brightness):
    return target_brightness / current_brightness

def get_average_brightness(image):
    grayscale_image = image.convert("L")
    histogram = grayscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    total_brightness = sum(i * w for i, w in enumerate(histogram))
    return total_brightness / pixels

def apply_dithering(image):
    return image.convert("P", palette=Image.ADAPTIVE, colors=256).convert("RGB")

def apply_noise_reduction(image, strength):
    return image.filter(ImageFilter.GaussianBlur(radius=strength))

def apply_gradient_smoothing(image, strength):
    return image.filter(ImageFilter.SMOOTH_MORE if strength > 1 else ImageFilter.SMOOTH)

def blend_images(image1, image2, alpha):
    return Image.blend(image1, image2, alpha)

class Deflicker:
    DESCRIPTION = "Experimental high-level deflicker pass for video/animation. Analyses brightness across neighbouring frames and blends, denoises and smooths gradients to reduce global flicker."
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "context_length": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "brightness_threshold": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01}),
                "blending_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_reduction_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "gradient_smoothing_strength": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "deflicker"
    CATEGORY = "SuperBeastsAI/Animation"

    def deflicker(self, images, context_length=5, brightness_threshold=0.05, blending_strength=0.5,
                  noise_reduction_strength=1.0, gradient_smoothing_strength=1, batch_size=10):
        num_frames = len(images)
        adjusted_tensor = []

        for i in range(0, num_frames, batch_size):
            batch_images = images[i:i+batch_size]

            # Convert batch tensor to a list of PIL images
            pil_images = [tensor2pil(image) for image in batch_images]

            adjusted_images = []

            for j in range(len(pil_images)):
                current_image = pil_images[j]
                context_start = max(0, i + j - context_length // 2)
                context_end = min(num_frames, i + j + context_length // 2 + 1)
                context_images = images[context_start:context_end]

                current_brightness = get_average_brightness(current_image)
                context_brightnesses = [get_average_brightness(tensor2pil(img)) for img in context_images]
                average_brightness = np.mean(context_brightnesses)

                if abs(current_brightness - average_brightness) > brightness_threshold:
                    brightness_factor = calculate_brightness_factor(average_brightness, current_brightness)
                    adjusted_image = adjust_brightness(current_image, brightness_factor)
                else:
                    adjusted_image = current_image

                # Apply noise reduction to the adjusted image
                denoised_image = apply_noise_reduction(adjusted_image, noise_reduction_strength)

                # Apply gradient smoothing to the denoised image
                smoothed_image = apply_gradient_smoothing(denoised_image, gradient_smoothing_strength)

                # Apply dithering to the smoothed image
                dithered_image = apply_dithering(smoothed_image)

                # Blend the dithered image with the original image using adaptive blending
                blending_alpha = min(1.0, blending_strength * (1.0 + abs(current_brightness - average_brightness)))
                blended_image = blend_images(current_image, dithered_image, blending_alpha)

                adjusted_images.append(blended_image)

            # Convert the adjusted PIL images back to a tensor
            adjusted_batch_tensor = torch.cat([pil2tensor(img) for img in adjusted_images], dim=0)
            adjusted_tensor.append(adjusted_batch_tensor)

        # Concatenate the adjusted batches along the first dimension
        adjusted_tensor = torch.cat(adjusted_tensor, dim=0)

        return (adjusted_tensor,)


def temporal_smoothing(frames, window_size):
    num_frames = len(frames)
    smoothed_frames = []

    for i in range(num_frames):
        start = max(0, i - window_size // 2)
        end = min(num_frames, i + window_size // 2 + 1)
        window_frames = frames[start:end]

        smoothed_frame = np.mean(window_frames, axis=0)
        smoothed_frames.append(smoothed_frame)

    return smoothed_frames

class PixelDeflicker:
    DESCRIPTION = "Experimental per-pixel temporal smoothing for animation. Operates in a sliding window to average noisy pixels while preserving detail, helping mitigate small-scale flicker."
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "blending_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pixelDeflicker"

    CATEGORY = "SuperBeastsAI/Animation"

    def pixelDeflicker(self, images, window_size=5, blending_strength=0.5, batch_size=10):
        num_frames = len(images)
        blended_tensor = []

        for i in range(0, num_frames, batch_size):
            batch_images = images[i:i+batch_size]

            # Convert batch tensor to a list of PIL images
            pil_images = [tensor2pil(image) for image in batch_images]

            # Convert PIL images to numpy arrays
            numpy_frames = [np.array(img) / 255.0 for img in pil_images]

            # Apply temporal smoothing to the numpy frames
            smoothed_frames = temporal_smoothing(numpy_frames, window_size)

            # Blend the smoothed frames with the original frames
            blended_frames = [
                np.clip(original * (1 - blending_strength) + smoothed * blending_strength, 0, 1)
                for original, smoothed in zip(numpy_frames, smoothed_frames)
            ]

            # Convert the blended frames back to PIL images
            blended_pil_images = [Image.fromarray((frame * 255).astype(np.uint8)) for frame in blended_frames]

            # Convert the blended PIL images back to a tensor
            blended_batch_tensor = torch.cat([pil2tensor(img) for img in blended_pil_images], dim=0)

            blended_tensor.append(blended_batch_tensor)

        # Concatenate the blended batches along the first dimension
        blended_tensor = torch.cat(blended_tensor, dim=0)

        return (blended_tensor,)

def resize_and_crop(pil_img, target_width, target_height):
    """Resize and crop an image to fit exactly the specified dimensions."""
    original_width, original_height = pil_img.size
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if target_aspect_ratio > aspect_ratio:
        # Target is wider than the image
        scale_factor = target_width / original_width
        scaled_height = int(original_height * scale_factor)
        scaled_width = target_width
    else:
        # Target is taller than the image
        scale_factor = target_height / original_height
        scaled_height = target_height
        scaled_width = int(original_width * scale_factor)

    # Resize the image
    resized_img = pil_img.resize((scaled_width, scaled_height), Image.BILINEAR)

    # Crop the image
    if scaled_width != target_width or scaled_height != target_height:
        left = (scaled_width - target_width) // 2
        top = (scaled_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        cropped_img = resized_img.crop((left, top, right, bottom))
    else:
        cropped_img = resized_img

    return cropped_img

class ImageBatchManagement:
    DESCRIPTION = "Resize, crop, limit, optionally shuffle or manually reorder an image batch so all frames match the specified resolution and sequence order. Returns the new batch and a CSV string listing filenames."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "order": 1}),
                "height": ("INT", {"default": 768}),
                "max_images": ("INT", {"default": 10}),  # New INT input for maximum number of images
                "random_order": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "new_manual_order": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "reorder"
    CATEGORY = "SuperBeastsAI/Image"

    def reorder(self, width, height, random_order, max_images, **kwargs):
        images = [kwargs["image1"]]  # Start with the required image1 input

        i = 2
        while f"image{i}" in kwargs:
            images.append(kwargs[f"image{i}"])
            i += 1

        if max_images is not None:
            images = images[:max_images]

        # Default order_output if new_manual_order isn't provided or is empty
        order_output = ",".join(str(idx + 1) for idx in range(len(images)))

        # Retrieve and apply new_manual_order if it exists
        if 'new_manual_order' in kwargs and kwargs['new_manual_order']:
            order_indices = [int(idx) - 1 for idx in kwargs['new_manual_order'].split(',') if idx.strip()]
            images = [images[idx] for idx in order_indices if idx < len(images)]
            order_output = kwargs['new_manual_order']

        processed_images = []
        for img in images:
            pil_img = tensor2pil(img)
            resized_cropped_img = resize_and_crop(pil_img, width, height)
            img_tensor = pil2tensor(resized_cropped_img)
            processed_images.append(img_tensor)

        result = torch.cat(processed_images, dim=0) if processed_images else torch.empty(0, 3, height, width)
        return (result, order_output)


class MaskBatchManagement:
    DESCRIPTION = "Resize, crop and re-order a batch of masks so they all share the same dimensions, or to match a new manual order list."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 768})
            },
            "optional": {
                "new_order": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "append"
    CATEGORY = "SuperBeastsAI/Masks"

    def append(self, width, height, new_order, **kwargs):
        masks = [kwargs["mask1"]]  # Start with the required mask1 input

        i = 2
        while f"mask{i}" in kwargs:
            masks.append(kwargs[f"mask{i}"])
            i += 1

        if new_order:
            order_indices = [int(idx) - 1 for idx in new_order.split(',') if idx.strip()]
            masks = [masks[idx] for idx in order_indices if idx < len(masks)]

        processed_masks = []
        for mask in masks:
            pil_mask = tensor2pil(mask)
            resized_cropped_mask = resize_and_crop(pil_mask, width, height)
            mask_tensor = pil2tensor(resized_cropped_mask)
            processed_masks.append(mask_tensor)

        result = torch.cat(processed_masks, dim=0) if processed_masks else torch.empty(0, 1, height, width)

        return (result,)

class StringListManager:
    DESCRIPTION = "Re-order or time-expand a list of strings (e.g. prompts or captions) so each item can repeat for N frames. Outputs the adjusted list as a single newline-separated string."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames_per_image": ("INT", {"default": 1, "min": 1, "step": 1})
            },
            "optional": {
                "new_order": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "reorder_strings"
    CATEGORY = "SuperBeastsAI/Utils"

    def reorder_strings(self, frames_per_image, new_order, **kwargs):
        strings = [kwargs["string1"]]  # Start with the required string1 input

        i = 2
        while f"string{i}" in kwargs:
            strings.append(kwargs[f"string{i}"])
            i += 1

        if new_order:
            order_indices = [int(idx) - 1 for idx in new_order.split(',') if idx.strip()]
            strings = [strings[idx] for idx in order_indices if idx < len(strings)]

        result = []
        for i, string in enumerate(strings):
            result.append('"{frames}": "{string}"'.format(frames=frames_per_image * i, string=string))

        return (",\n".join(result),)

class SuperPopColorAdjustment:
    """Generate a series of color-adjusted images by blending the residual produced by a model-based correction with the original image at different strengths.

    The node calls the provided *SBModel* once to obtain a fully-corrected reference image (Requires use of "SB Load Model" node).  It then blends that result back into the
    original image `count` times using evenly spaced strength values between `max_strength/count` and `max_strength` (inclusive).
    """
    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SBMODEL", ),
                "image": ("IMAGE", ),
                "max_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "count": ("INT", {"default": 1, "min": 1, "max": 99999, "step": 1}),
                "overlap": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01}),
                "initial_context_for_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "context": ("IMAGE", ),  # Optional context image(s)
            }
        }

    # primary output: processed image batch
    # secondary output: per-frame filename prefix strings for Save Image
    # third output: raw residual tensors (type SPCA_RESIDUAL) so that a
    # dedicated blend node can re-apply any arbitrary strength later on.
    RETURN_TYPES = ("IMAGE", "STRING", "SPCA_RESIDUAL")
    RETURN_NAMES = ("images", "filename_prefix", "residuals")
    FUNCTION = "apply_adjustment"
    CATEGORY = "SuperBeastsAI/Image"

    def _run_model(self, sb_model_info, pil_image, ctx_np=None):
        """Attempt to run the supplied SBModel on *pil_image* to obtain a corrected image.
        The implementation gracefully falls back to a PIL colour enhancement if the model
        does not expose a compatible callable interface yet.  This allows the node to work
        out-of-the-box while the dedicated SuperBeasts model loader is still under development."""
        try:
            # Build context thumbnail from global image once if not provided
            if ctx_np is None:
                thumb = pil_image.resize((CONTEXT_THUMBNAIL_SIZE[1], CONTEXT_THUMBNAIL_SIZE[0]), Image.Resampling.BILINEAR)
                ctx_np_local = _add_extra_color_channels_np(np.asarray(thumb, dtype=np.float32) / 255.0)
            else:
                ctx_np_local = ctx_np

            # --- Torch path ---
            if isinstance(sb_model_info, dict) and sb_model_info.get("type") == "torch":
                import torch
                model_pt = sb_model_info["model"]
                device_str = sb_model_info["device"]

                # Prepare inputs NCHW float32
                tile_np = _add_extra_color_channels_np(np.asarray(pil_image, dtype=np.float32) / 255.0)

                tile_t = torch.from_numpy(tile_np.transpose(2,0,1)).unsqueeze(0).to(device_str)
                ctx_t = torch.from_numpy(ctx_np_local.transpose(2,0,1)).unsqueeze(0).to(device_str)

                with torch.no_grad():
                    residual = model_pt(tile_t, ctx_t)
                residual_np = residual.squeeze(0).cpu().numpy().transpose(1,2,0)

                tile_base = np.asarray(pil_image, dtype=np.float32) / 255.0
                corrected_np = np.clip(tile_base + residual_np, 0.0, 1.0)
                result = Image.fromarray((corrected_np * 255.0).astype(np.uint8))

            # --- ONNX path ---
            elif isinstance(sb_model_info, dict) and sb_model_info.get("type") == "onnx":
                import onnxruntime as ort  # type: ignore
                session: ort.InferenceSession = sb_model_info["session"]
                input_names: List[str] = sb_model_info["input_names"]

                # Prepare inputs (NHWC float32)
                ctx_np_batch = ctx_np_local[np.newaxis, ...]  # add batch dim

                tile_np = _add_extra_color_channels_np(np.asarray(pil_image, dtype=np.float32) / 255.0)
                tile_np = tile_np[np.newaxis, ...]

                # Some exported models keep NHWC; others are converted to NCHW. Detect by checking input shape channels position.
                def maybe_transpose(inp: np.ndarray, shape) -> np.ndarray:
                    # shape may have None for batch, use length check
                    if len(shape) == 4 and shape[1] == 3:  # likely NCHW
                        return np.transpose(inp, (0, 3, 1, 2))
                    return inp  # assume NHWC

                tile_ready = maybe_transpose(tile_np, session.get_inputs()[0].shape)
                ctx_ready = maybe_transpose(ctx_np_batch, session.get_inputs()[1].shape)

                ort_inputs = {
                    input_names[0]: tile_ready.astype(np.float32),
                    input_names[1]: ctx_ready.astype(np.float32),
                }

                pred_residual = session.run(None, ort_inputs)[0]
                if pred_residual.shape[1] == 3 and pred_residual.ndim == 4:  # NCHW output
                    pred_residual = np.transpose(pred_residual, (0, 2, 3, 1))
                residual_np = pred_residual[0]

                tile_base = np.asarray(pil_image, dtype=np.float32) / 255.0
                corrected_np = np.clip(tile_base + residual_np, 0.0, 1.0)
                result = Image.fromarray((corrected_np * 255.0).astype(np.uint8))

            else:
                # If the supplied model information structure is not recognised, raise an error so
                # users are immediately aware of the mis-configuration instead of falling back.
                raise TypeError("Unsupported SBMODEL object – expected dict with type 'torch' or 'onnx'.")

            # Convert potential tensor outputs to PIL
            try:
                import torch as _torch
            except ImportError:
                _torch = None

            if _torch is not None and isinstance(result, _torch.Tensor):
                result = tensor2pil(result)
            if not isinstance(result, Image.Image):
                raise TypeError("Model output is not an image")
            return result
        except Exception:
            # Re-raise any exception so the UI surfaces the real failure rather than silently
            # producing an uncorrected image.
            raise

    def _get_patch_positions(self, img_len: int, patch_len: int, overlap_px: int):
        """Return list of starting indices so patches cover dimension with given overlap."""
        if patch_len >= img_len:
            return [0]
        stride = max(1, patch_len - overlap_px)
        positions = [0]
        while True:
            next_pos = positions[-1] + stride
            if next_pos + patch_len >= img_len:
                break
            positions.append(next_pos)
        last_start = img_len - patch_len
        if positions[-1] != last_start:
            positions.append(last_start)
        return positions

    # --- Patch weight map (mirrors extras/inference_utils._create_patch_weight_map) ---
    def _create_patch_weight_map(self, ph: int, pw: int, overlap: int | tuple[int,int]):
        if isinstance(overlap, (list, tuple)) and len(overlap) == 2:
            oh, ow = int(overlap[0]), int(overlap[1])
        else:
            oh = ow = int(overlap)

        oh = max(0, min(oh, ph))
        ow = max(0, min(ow, pw))

        if oh == 0 and ow == 0:
            return np.ones((ph, pw), dtype=np.float32)

        w = np.ones((ph, pw), dtype=np.float32)
        if oh > 0:
            for i in range(oh):
                fade = (i + 1) / (oh + 1)
                w[i, :] *= fade
                w[ph - 1 - i, :] *= fade
        if ow > 0:
            for i in range(ow):
                fade = (i + 1) / (ow + 1)
                w[:, i] *= fade
                w[:, pw - 1 - i] *= fade
        return w

    def apply_adjustment(self, model, image, max_strength=1.0, count=1, overlap=0.5,
                         initial_context_for_batch=False, context=None):
        # Ensure *image* is a batch tensor – iterate over each frame
        output_tensors: list[torch.Tensor] = []
        filename_prefixes: list[str] = []
        residual_tensors: list[torch.Tensor] = []

        # Cache for the first computed context thumbnail when reusing across batch
        ctx_np_cached = None

        # Helper to obtain the source PIL for context, considering optional input
        def _get_context_pil(idx, orig_pil):
            if context is not None:
                # Guard against length mismatch – fallback to first element if idx out of range
                ctx_tensor = context[idx] if idx < context.shape[0] else context[0]
                return tensor2pil(ctx_tensor)
            # Default: use the original image as its own context
            return orig_pil

        pbar = None  # progress bar placeholder
        try:
            import importlib
            ProgressBar = getattr(importlib.import_module("comfy.utils"), "ProgressBar", None)  # type: ignore[attr-defined,ignored-type]
            if ProgressBar is not None:
                pbar = ProgressBar(image.shape[0])
        except Exception:
            pbar = None

        for idx, img_tensor in enumerate(image):
            # Convert to PIL for easier colour operations
            original_pil = tensor2pil(img_tensor)

            # -------------------------------------------------
            # Prepare / reuse context thumbnail
            # -------------------------------------------------
            if initial_context_for_batch and ctx_np_cached is not None:
                ctx_np_global = ctx_np_cached
            else:
                ctx_source_pil = _get_context_pil(idx, original_pil)
                # Resize exactly to CONTEXT_THUMBNAIL_SIZE using bicubic resampling
                thumb_global = ctx_source_pil.resize((CONTEXT_THUMBNAIL_SIZE[1], CONTEXT_THUMBNAIL_SIZE[0]),
                                                    Image.Resampling.BICUBIC)
                ctx_np_global = _add_extra_color_channels_np(np.asarray(thumb_global, dtype=np.float32) / 255.0)

                if initial_context_for_batch:
                    ctx_np_cached = ctx_np_global

            # ---------- Patch-based processing ----------
            h, w = original_pil.size[1], original_pil.size[0]
            patch_size = model.get("patch_size", 512) if isinstance(model, dict) else 512
            patch_size = int(patch_size)
            overlap_px = int(patch_size * overlap)
            if overlap_px >= patch_size:
                overlap_px = patch_size // 2

            # Prepare accumulators for residual image
            residual_acc = np.zeros((h, w, 3), dtype=np.float32)
            counter = np.zeros((h, w, 1), dtype=np.float32)

            x_positions = self._get_patch_positions(w, patch_size, overlap_px)
            y_positions = self._get_patch_positions(h, patch_size, overlap_px)

            orig_np_full = np.array(original_pil, dtype=np.float32) / 255.0

            # ctx_np_global already prepared above

            for y in y_positions:
                for x in x_positions:
                    patch = original_pil.crop((x, y, x + patch_size, y + patch_size))
                    patch_corrected = self._run_model(model, patch, ctx_np_global)

                    # Resize patch_corrected to patch size in case model changes size
                    if patch_corrected.size != (patch_size, patch_size):
                        patch_corrected = patch_corrected.resize((patch_size, patch_size), Image.Resampling.BILINEAR)

                    orig_patch_np = np.array(patch, dtype=np.float32) / 255.0
                    corr_patch_np = np.array(patch_corrected, dtype=np.float32) / 255.0

                    # Determine overlap area within the original image boundaries
                    target_h = min(patch_size, h - y)
                    target_w = min(patch_size, w - x)

                    # If the patch is already full size use it directly -----------------------
                    if target_h == patch_size and target_w == patch_size:
                        residual_patch = corr_patch_np - orig_patch_np
                    else:
                        # --------------------------------------------------------------
                        # Pad the original patch to 512×512 instead of scaling to avoid
                        # introducing blur / ghosting. Use edge padding.
                        # --------------------------------------------------------------
                        # Crop to the valid region before padding
                        orig_cropped = orig_patch_np[:target_h, :target_w, :]

                        # --- Symmetric edge-padding approach (avoid stretched background artefacts) ---
                        pad_vert = patch_size - target_h
                        pad_horz = patch_size - target_w
                        pad_top = pad_vert // 2
                        pad_bottom = pad_vert - pad_top
                        pad_left = pad_horz // 2
                        pad_right = pad_horz - pad_left

                        # Edge-pad the crop up to (patch_size, patch_size)
                        orig_pad = np.pad(orig_cropped, (
                            (pad_top, pad_bottom),
                            (pad_left, pad_right),
                            (0, 0)), mode='edge')

                        # Convert to PIL for model inference
                        pad_pil = Image.fromarray((orig_pad * 255.0).astype(np.uint8))

                        # Run model on padded patch – override earlier corr_patch_np
                        patch_corrected = self._run_model(model, pad_pil, ctx_np_global)

                        if patch_corrected.size != (patch_size, patch_size):
                            patch_corrected = patch_corrected.resize((patch_size, patch_size), Image.Resampling.BILINEAR)

                        corrected_pad_np = np.array(patch_corrected, dtype=np.float32) / 255.0

                        # Crop back to original region
                        corrected_crop = corrected_pad_np[pad_top:pad_top + target_h, pad_left:pad_left + target_w, :]

                        # Residual relative to original crop
                        residual_patch = corrected_crop - orig_cropped

                    # Weight map for smooth blending (same crop)
                    weight_full = self._create_patch_weight_map(patch_size, patch_size, overlap_px)
                    if target_h != patch_size or target_w != patch_size:
                        weight = weight_full[pad_top:pad_top + target_h, pad_left:pad_left + target_w]
                    else:
                        weight = weight_full[:target_h, :target_w]

                    # Determine the actual location in the full image where this residual belongs
                    residual_acc[y:y + target_h, x:x + target_w, :] += residual_patch * weight[:, :, np.newaxis]
                    counter[y:y + target_h, x:x + target_w, :] += weight[:, :, np.newaxis]

            # Use a tiny epsilon instead of 1.0 so that edge pixels retain their full residual
            counter = np.maximum(counter, 1e-6)
            residual_full = residual_acc / counter

            # Generate blended outputs for this frame
            strengths = [max_strength * (count - i) / count for i in range(count)]
            for s in strengths:
                blended_np = np.clip(orig_np_full + residual_full * s, 0.0, 1.0)
                blended_pil = Image.fromarray((blended_np * 255.0).astype(np.uint8))

                # Convert blended PIL → tensor and append to batch list
                output_tensors.append(pil2tensor(blended_pil))

                # Build filename prefix string per‐image (we will join later)
                filename_prefixes.append(f"SPCA_{s:.2f}_")

                # Store residual for this output so downstream nodes can re-blend
                residual_tensor = torch.from_numpy(residual_full).unsqueeze(0).float()
                residual_tensors.append(residual_tensor)
            # Update progress bar
            if pbar is not None:
                pbar.update_absolute(idx + 1)

        # ------------------------------------------------------------------
        # Finalise outputs – convert accumulated lists to batched tensors and
        # join filename prefixes into a single string (Save Image expects str)
        # ------------------------------------------------------------------

        if not output_tensors:
            # Fallback – no processing happened; return originals and zero residual.
            images_batch = image
            residual_batch = [torch.zeros_like(image)]
            filename_prefix_str = ""
        else:
            images_batch = torch.cat(output_tensors, dim=0)
            residual_batch = residual_tensors  # keep list per-frame
            filename_prefix_str = "".join(filename_prefixes)

        return (images_batch, filename_prefix_str, residual_batch)

class SuperPopResidualBlend:
    """Interactively re-apply a residual predicted by **Super Pop Color Adjustment** at any arbitrary strength.

    Inputs
    ------
    image      : original image batch to adjust
    residual   : batch of residual tensors output from SuperPopColorAdjustment
    strength   : scalar multiplier (e.g. 0.0-2.0) applied to the residual
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "residual": ("SPCA_RESIDUAL", ),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filename_prefix")
    FUNCTION = "blend"
    CATEGORY = "SuperBeastsAI/Image"

    def blend(self, image, residual, strength=1.0):
        """Apply *residual × strength* to *image* per-frame."""
        # Ensure each input is a list-like batch for easy zip processing.
        if not isinstance(image, (list, tuple)):
            image = [image]
        if not isinstance(residual, (list, tuple)):
            residual = [residual]

        if len(residual) != len(image):
            # Simple broadcast: repeat the first residual for all images if counts mismatch
            residual = residual * len(image)

        blended_out: list[np.ndarray] = []
        prefixes: list[str] = []
        for img_tensor, res_tensor in zip(image, residual):
            # Tensors → numpy, remove any leading singleton batch dim so PIL
            # later sees a plain (H, W, 3) array.
            img_np = img_tensor.squeeze().cpu().numpy()
            res_np = res_tensor.squeeze().cpu().numpy()

            blended_np = np.clip(img_np + res_np * strength, 0.0, 1.0)
            # Convert to 0-255 uint8 so downstream Save-Image & comparer nodes
            # don’t need to rescale.
            blended_uint8 = (blended_np * 255.0).astype(np.uint8)
            blended_out.append(np.squeeze(blended_uint8))
            prefixes.append(f"SPCA_{strength:.2f}_")

        # Flatten arbitrarily nested lists/tuples
        def _flatten(seq):
            for elem in seq:
                if isinstance(elem, (list, tuple)):
                    yield from _flatten(elem)
                else:
                    yield elem

        flat_imgs: list[np.ndarray] = list(_flatten(blended_out))

        # Ensure each image is numpy array (H, W, 3)
        clean_imgs: list[np.ndarray] = []
        for a in flat_imgs:
            arr = np.asarray(a)
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            while arr.ndim > 3:
                arr = np.squeeze(arr)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            clean_imgs.append(arr)

        # Convert numpy images → torch tensors expected by ComfyUI
        tensors = [pil2tensor(Image.fromarray(img)) for img in clean_imgs]
        batch_tensor = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

        # Provide a single filename_prefix string.
        joined_prefix = "".join(prefixes)

        return (batch_tensor, joined_prefix)

NODE_CLASS_MAPPINGS = {
    'HDR Effects (SuperBeasts.AI)': HDREffects,
    'Make Resized Mask Batch (SuperBeasts.AI)': MakeResizedMaskBatch,
    'Mask Batch Manager (SuperBeasts.AI)': MaskBatchManagement,
    'Image Batch Manager (SuperBeasts.AI)': ImageBatchManagement,
    'String List Manager (SuperBeasts.AI)': StringListManager,
    'Deflicker - Experimental (SuperBeasts.AI)': Deflicker,
    'Pixel Deflicker - Experimental (SuperBeasts.AI)': PixelDeflicker,
    # Legacy key kept for compatibility – new branded name below
    'Super Color Adjustment (SuperBeasts.AI)': SuperPopColorAdjustment,
    'Super Pop Color Adjustment (SuperBeasts.AI)': SuperPopColorAdjustment,
    'SB Load Model (SuperBeasts.AI)': SBLoadModel,
    'Super Pop Residual Blend (SuperBeasts.AI)': SuperPopResidualBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'HDREffects': 'HDR Effects (SuperBeasts.AI)',
    'MakeResizedMaskBatch': 'Make Resized Mask Batch (SuperBeasts.AI)',
    'MaskBatchManagement':'Mask Batch Manager (SuperBeasts.AI)',
    'ImageBatchManagement':'Image Batch Manager (SuperBeasts.AI)',
    'StringListManager': 'String List Manager (SuperBeasts.AI)',
    'Deflicker': 'Deflicker - Experimental (SuperBeasts.AI)',
    'PixelDeflicker': 'Pixel Deflicker - Experimental (SuperBeasts.AI)',
    'SuperPopColorAdjustment': 'Super Pop Color Adjustment (SuperBeasts.AI)',
    'SBLoadModel': 'SB Load Model (SuperBeasts.AI)',
    'SuperPopResidualBlend': 'Super Pop Residual Blend (SuperBeasts.AI)',
}