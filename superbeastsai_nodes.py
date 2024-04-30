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

class MakeResizedMaskBatch:
    """
    Creates a batch of masks from multiple individual masks or batches.
    """
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
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 768}),
                "ordering_enabled": (["disabled", "enabled"], {"default": "disabled"}),
                "image1": ("IMAGE",)  # Ensure at least one image is required
            },
            "optional": {
                "new_order": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reorder"
    CATEGORY = "SuperBeastsAI/Image"

    def reorder(self, width, height, ordering_enabled, new_order, **kwargs):
        images = [kwargs["image1"]]  # Start with the required image1 input

        i = 2
        while f"image{i}" in kwargs:
            images.append(kwargs[f"image{i}"])
            i += 1

        if ordering_enabled == "enabled" and new_order:
            order_indices = [int(idx) - 1 for idx in new_order.split(',') if idx.strip()]
            images = [images[idx] for idx in order_indices if idx < len(images)]

        processed_images = []
        for img in images:
            pil_img = tensor2pil(img)
            resized_cropped_img = resize_and_crop(pil_img, width, height)
            img_tensor = pil2tensor(resized_cropped_img)
            processed_images.append(img_tensor)

        result = torch.cat(processed_images, dim=0) if processed_images else torch.empty(0, 3, height, width)
        return (result,)

class MaskBatchManagement:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 768}),
                "ordering_enabled": (["disabled", "enabled"], {"default": "disabled"}),
                "mask1": ("MASK",),
            },
            "optional": {
                "new_order": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "append"
    CATEGORY = "SuperBeastsAI/Masks"

    def append(self, width, height, ordering_enabled, new_order, **kwargs):
        masks = [kwargs["mask1"]]  # Start with the required mask1 input

        i = 2
        while f"mask{i}" in kwargs:
            masks.append(kwargs[f"mask{i}"])
            i += 1

        if ordering_enabled == "enabled" and new_order:
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



NODE_CLASS_MAPPINGS = {
    'HDR Effects (SuperBeasts.AI)': HDREffects,
    'Make Resized Mask Batch (SuperBeasts.AI)': MakeResizedMaskBatch,
    'Deflicker (SuperBeasts.AI)': Deflicker,
    'Pixel Deflicker (SuperBeasts.AI)': PixelDeflicker,
    'Mask Batch Manager (SuperBeasts.AI)': MaskBatchManagement,
    'Image Batch Manager (SuperBeasts.AI)': ImageBatchManagement   
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'HDREffects': 'HDR Effects (SuperBeasts.AI)',
    'MakeResizedMaskBatch': 'Make Resized Mask Batch (SuperBeasts.AI)',
    'Deflicker': 'Deflicker (SuperBeasts.AI)',
    'PixelDeflicker': 'Pixel Deflicker (SuperBeasts.AI)',
    'MaskBatchManagement':'Mask Batch Manager (SuperBeasts.AI)',
    'ImageBatchManagement':'Image Batch Manager (SuperBeasts.AI)'
}