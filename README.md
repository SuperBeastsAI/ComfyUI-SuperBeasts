# SuperBeastsAI Nodes

This repository contains custom nodes for ComfyUI created and used by SuperBeasts.AI (@SuperBeasts.AI on Instagram)

## Updates

- 31/07/24: Resolved bugs with dynamic input thanks to @Amorano. Add Github Action for Publishing to Comfy Registry thanks to @haohaocreates
- 30/07/24: Moved Deflicker & PixelDeflicker to Experimental labels (this will require readding them in your WF but I wanted this to be clearer)
- 30/04/24: 
- - Updated readme with documentation
- - Added Deflicker and PixelDeflicker nodes for reducing flickering artifacts in image sequences. 
- - Introduced CrossFadeImageBatches node for smooth cross-fade transitions between image batches
- - Enhanced ImageBatchManagement and MaskBatchManagement nodes with resizing, cropping, and reordering capabilities
- 27/03/24: Released and also pushed an update to resolve batched images for videos etc

# Image Effects

## HDR Effects (SuperBeasts.AI)

The HDR Effects is an image processing application that enhances the dynamic range and visual appeal of input images. It provides a set of adjustable parameters to fine-tune the HDR effect according to user preferences.

### Features

- Adjusts the intensity of shadows, highlights, and overall HDR effect
- Applies gamma correction to control the overall brightness and contrast
- Enhances contrast and color saturation for more vibrant results
- Preserves color accuracy by processing the image in the LAB color space
- Utilizes luminance-based masks for targeted adjustments
- Blends the adjusted luminance with the original luminance for a balanced effect

## Examples

![Example 1](examples/ex1.png)


![Example 2](examples/ex2.png)


### Parameter details

The application provides the following adjustable parameters:

- `hdr_intensity` (default: 0.5, range: 0.0 to 5.0, step: 0.01):
  - Controls the overall intensity of the HDR effect
  - Higher values result in a more pronounced HDR effect

- `shadow_intensity` (default: 0.25, range: 0.0 to 1.0, step: 0.01):
  - Adjusts the intensity of shadows in the image
  - Higher values darken the shadows and increase contrast

- `highlight_intensity` (default: 0.75, range: 0.0 to 1.0, step: 0.01):
  - Adjusts the intensity of highlights in the image
  - Higher values brighten the highlights and increase contrast

- `gamma_intensity` (default: 0.25, range: 0.0 to 1.0, step: 0.01):
  - Controls the gamma correction applied to the image
  - Higher values increase the overall brightness and contrast

- `contrast` (default: 0.1, range: 0.0 to 1.0, step: 0.01):
  - Enhances the contrast of the image
  - Higher values result in more pronounced contrast

- `enhance_color` (default: 0.25, range: 0.0 to 1.0, step: 0.01):
  - Enhances the color saturation of the image
  - Higher values result in more vibrant colors
 

# Image & Mask Batch Management

Keep your image and masks sized, cropped and ordered how ever you like without having to recreate the masks or mess with connections. 
Note the resizing capability doesn't provide any settings, we simply crop and resize to maximise the image size with your provided width/height. 

![Batch Management](examples/BatchManagers.jpg)


## ImageBatchManagement

The ImageBatchManagement node provides functionality to resize, crop, and reorder a batch of images. It ensures that all images in the batch have the same specified dimensions.

### Features
- Resizes and crops images to the specified width and height
- Allows reordering the images in the batch based on a provided order
- Supports unlimited input images

## MaskBatchManagement

The MaskBatchManagement node is similar to the ImageBatchManagement node but works with mask inputs. It resizes, crops, and reorders a batch of masks to match the specified dimensions.

### Features
- Resizes and crops masks to the specified width and height
- Allows reordering the masks in the batch based on a provided order
- Supports unlimited input masks


# Deprecated / Removed

## Deflicker & PixelDeflicker

The Deflicker nodes have been marked as "Experimental" as they were experiments that were released but not really fit for purpose / doing what you would expect of a deflicker node. 
At some point I will be removing these nodes as I won't be supporting them at this stage.  

## MakeResizedMaskBatch (Deprecated please use MaskBatchManagement)
The MakeResizedMaskBatch node creates a batch of masks from multiple individual masks or batches. 
It resizes and crops the input masks to match the specified width and height.

## Cross Fade Image Batches (SuperBeasts.AI)
There is another preexisting node that does this and more so I've removed this function.

