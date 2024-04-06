import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageEnhance, ImageCms
from PIL.PngImagePlugin import PngInfo
import torch
import torch.nn.functional as F

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

NODE_CLASS_MAPPINGS = {
    'HDR Effects (SuperBeasts.AI)': HDREffects
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'HDREffects': 'HDR Effects (SuperBeasts.AI)'
}
