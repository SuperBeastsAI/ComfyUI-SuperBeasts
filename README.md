# SuperBeastsAI Nodes

This repository contains custom nodes for ComfyUI created and used by SuperBeasts.AI (@SuperBeasts.AI on Instagram)

## Updates
- 19/07/25: Super Pop Colour Adjustment model and node release
- 31/07/24: Resolved bugs with dynamic input thanks to @Amorano. Add Github Action for Publishing to Comfy Registry thanks to @haohaocreates
- 30/07/24: Moved Deflicker & PixelDeflicker to Experimental labels (this will require readding them in your WF but I wanted this to be clearer)
- 30/04/24: 
- - Updated readme with documentation
- - Added Deflicker and PixelDeflicker nodes for reducing flickering artifacts in image sequences. 
- - Introduced CrossFadeImageBatches node for smooth cross-fade transitions between image batches
- - Enhanced ImageBatchManagement and MaskBatchManagement nodes with resizing, cropping, and reordering capabilities
- 27/03/24: Released and also pushed an update to resolve batched images for videos etc

---
# Super Beasts Custom Models (and nodes)

![SPCA-Header](examples/SPCA-Header.webp)

## Super Pop Color Adjustment (SuperBeasts.AI)

Bring the **SuperBeasts “pop”** to any image in one click. Months of solo R&D, hundreds of hand‑tuned colour‑grade examples, and a custom residual CNN trained to push **vibrant palette, deep blacks, crisp highlights, and HDR bite** with ease.

### Why I built it
Everything I post gets graded: lift blacks, pull whites, bend curves, punch colour. Existing auto tools (and many AI outputs — looking at you, muddy yellow “whites”) just weren’t landing, especially across batches or iterative generations. Training my own lightweight correction model let me reclaim hours of post and lock in a consistent SuperBeasts look I could reuse — and share.

**Personal project, shared as‑is.**  
Works great across my SuperBeasts production flow and a wide mix of test art, but it’s *not* exhaustively validated on every style, lighting condition, or colour space. Expect edge cases, occasional overcorrection, and behaviour that varies with resolution/patch overlap. Please experiment and let me know what you think!

If you want to show support please take a momenet to follow me on Instagram <a href="https://www.instagram.com/SuperBeasts.AI" target="_blank">@SuperBeasts.AI</a>

Happy generating!!

### Using Super Pop

1. Add **SB Load Model (SuperBeasts.AI)** node.
2. Pick **SuperPopColorAdjustment/latest**.
3. Connect to **Super Pop Color Adjustment (SuperBeasts.AI)** node.
4. Input source image and tune *Max Strength*, *Count*, *Overlap* (recommendations below)

**Auto model weight download:**
Weights are automatically downloaded the first time the **SB Load Model** node runs using a specific version or 'latest'.
If you prefer manual download, grab them from the GitHub release page:
`https://github.com/SuperBeastsAI/SuperBeastsAI-Models/tree/main/SuperPopColorAdjustment/`  
and drop the `.onnx` file into `custom_nodes/ComfyUI-SuperBeasts/models/`.

**License:**
Weights are SPCA-Community-NoSaaS. Local/client use OK; no public SaaS redistribution. <a href="https://github.com/SuperBeastsAI/SuperBeastsAI-Models/blob/main/SuperPopColorAdjustment/LICENSE.txt" target="_blank">See model repo</a>.


<h2>Same settings multiple situations corrected</h2>

<h3>Highlights/Shadows Correction</h3>

<table width="100%">
  <tr>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/1a.png" width="100%" alt="Original 1"></td>
    <td><img src="examples/SPCA/1b.png" width="100%" alt="S.P.C.A 1"></td>
    <td><img src="examples/SPCA/5a.png" width="100%" alt="Original 2"></td>
    <td><img src="examples/SPCA/5b.png" width="100%" alt="S.P.C.A 2"></td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/2a.png" width="100%" alt="Original 3"></td>
    <td><img src="examples/SPCA/2b.png" width="100%" alt="S.P.C.A 3"></td>
    <td><img src="examples/SPCA/4a.png" width="100%" alt="Original 4"></td>
    <td><img src="examples/SPCA/4b.png" width="100%" alt="S.P.C.A 4"></td>
  </tr>
</table>

<h3>Dull/Washed Out/Muted Correction</h3>

<table width="100%">
  <tr>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/41a.png" width="100%" alt="Original 1"></td>
    <td><img src="examples/SPCA/41b.png" width="100%" alt="S.P.C.A 1"></td>
    <td><img src="examples/SPCA/16a.png" width="100%" alt="Original 2"></td>
    <td><img src="examples/SPCA/16b.png" width="100%" alt="S.P.C.A 2"></td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/27a.png" width="100%" alt="Original 3"></td>
    <td><img src="examples/SPCA/27b.png" width="100%" alt="S.P.C.A 3"></td>
    <td><img src="examples/SPCA/24a.png" width="100%" alt="Original 4"></td>
    <td><img src="examples/SPCA/24b.png" width="100%" alt="S.P.C.A 4"></td>
  </tr>
</table>


<h3>Black/White Levels & Contrast Correction</h3>

<table width="100%">
  <tr>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/6a.png" width="100%" alt="Original 1"></td>
    <td><img src="examples/SPCA/6b.png" width="100%" alt="S.P.C.A 1"></td>
    <td><img src="examples/SPCA/9a.png" width="100%" alt="Original 2"></td>
    <td><img src="examples/SPCA/9b.png" width="100%" alt="S.P.C.A 2"></td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/38a.png" width="100%" alt="Original 3"></td>
    <td><img src="examples/SPCA/38b.png" width="100%" alt="S.P.C.A 3"></td>
    <td><img src="examples/SPCA/26a.png" width="100%" alt="Original 4"></td>
    <td><img src="examples/SPCA/26b.png" width="100%" alt="S.P.C.A 4"></td>
  </tr>
</table>


<h3>Color Correction</h3>

<table width="100%">
  <tr>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/7a.png" width="100%" alt="Original 1"></td>
    <td><img src="examples/SPCA/7b.png" width="100%" alt="S.P.C.A 1"></td>
    <td><img src="examples/SPCA/14a.png" width="100%" alt="Original 2"></td>
    <td><img src="examples/SPCA/14b.png" width="100%" alt="S.P.C.A 2"></td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/13a.png" width="100%" alt="Original 3"></td>
    <td><img src="examples/SPCA/13b.png" width="100%" alt="S.P.C.A 3"></td>
    <td><img src="examples/SPCA/34a.png" width="100%" alt="Original 4"></td>
    <td><img src="examples/SPCA/34b.png" width="100%" alt="S.P.C.A 4"></td>
  </tr>
</table>


<h3>Vibrancy Improvements</h3>

<table width="100%">
  <tr>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/25a.png" width="100%" alt="Original 1"></td>
    <td><img src="examples/SPCA/25b.png" width="100%" alt="S.P.C.A 1"></td>
    <td><img src="examples/SPCA/21a.png" width="100%" alt="Original 2"></td>
    <td><img src="examples/SPCA/21b.png" width="100%" alt="S.P.C.A 2"></td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/32a.png" width="100%" alt="Original 3"></td>
    <td><img src="examples/SPCA/32b.png" width="100%" alt="S.P.C.A 3"></td>
    <td><img src="examples/SPCA/28a.png" width="100%" alt="Original 4"></td>
    <td><img src="examples/SPCA/28b.png" width="100%" alt="S.P.C.A 4"></td>
  </tr>
</table>

<h3>Restoration</h3>
<p>Potential uses in old photo restorations and corrections.</p>

<table width="100%">
  <tr>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/18a.png" width="100%" alt="Original 1"></td>
    <td><img src="examples/SPCA/18b.png" width="100%" alt="S.P.C.A 1"></td>
    <td><img src="examples/SPCA/39a.png" width="100%" alt="Original 2"></td>
    <td><img src="examples/SPCA/39b.png" width="100%" alt="S.P.C.A 2"></td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/37a.png" width="100%" alt="Original 3"></td>
    <td><img src="examples/SPCA/37b.png" width="100%" alt="S.P.C.A 3"></td>
    <td><img src="examples/SPCA/40a.png" width="100%" alt="Original 4"></td>
    <td><img src="examples/SPCA/40b.png" width="100%" alt="S.P.C.A 4"></td>
  </tr>
</table>

<h3>Color balance correction</h3>

<table width="100%">
  <tr>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/18a.png" width="100%" alt="Original 1"></td>
    <td><img src="examples/SPCA/18b.png" width="100%" alt="S.P.C.A 1"></td>
    <td><img src="examples/SPCA/16a.png" width="100%" alt="Original 2"></td>
    <td><img src="examples/SPCA/16b.png" width="100%" alt="S.P.C.A 2"></td>
  </tr>
  <tr>
    <td><img src="examples/SPCA/16a.png" width="100%" alt="Original 3"></td>
    <td><img src="examples/SPCA/16b.png" width="100%" alt="S.P.C.A 3"></td>
    <td><img src="examples/SPCA/15a.png" width="100%" alt="Original 4"></td>
    <td><img src="examples/SPCA/15b.png" width="100%" alt="S.P.C.A 4"></td>
  </tr>
</table>


### How it works

1. **SB Load Model** node loads the ONNX-formatted colour-adjustment network
    (`models/SuperBeasts_ColorAdjustment_512px_V1.onnx`).
2. The model analyses the image in 512 × 512 patches, predicting a residual colour grade.
3. Patches are seamlessly stitched back with configurable **strength**, **count** (variant
    batching) and **overlap** for large-resolution output.
4. A 64 × 64 **context** thumbnail guides the global colour mapping – leave it empty for
    natural results or experiment with creative transfers.

### Recommended quick-start

| Parameter | Suggested | Notes |
|-----------|-----------|-------|
| Image | Image dimensions ideally <2048px | Techncially supports any size but due to 512px patch size local adjustments become increasingly obvious as size increases. |
| Max strength  | **1.0**   | 0.5-2.0 for subtle ➜ dramatic |
| Count     | 1         | Increase for automatic strength batching up to your max strength value. E.g. Max strength 1.0 with count 2.0 runs strength at 0.5 and 1.0 outputting 2 images |
| Overlap   | At least 0.3-0.4 if image is >512px  | Ideally up to 0.9 if GPU / Compute permits for maximum quality. |
| Context (Image) | Leave empty | See detailed notes on context below. |
| Initial context for batch | False | Use the first input images context for all images processed. If set to True this may be useful to reduce correction variation across the batch E.g. Video frames |

### Known limitations / quirks

* **Patch size sensitivity:** The network works on 512 px tiles – with images above ~2 K you may
  notice slight local exposure shifts.  Pushing *Overlap* up to 0.8-0.9 blends these out much better for higher quality results at the
  cost of extra compute/time.
* **Atmospheric overlays:** Because the model corrects blacks / whites aggressively it can
  dial back matte-style colour washes or cinematic tints.  Keep *Strength* around 0.7, or increase count for variation options or blend
  the result back with the original.
* **Very high strengths:** Values above 2.0 can clip highlights or oversaturate; use sparingly
  for stylised effects.

### Creative workflow ideas

1. **HDR Sandwich** – Run Super Pop, then feed the output into **HDR Effects** (with gentle
   settings) to amplify micro-contrast.
3. **Colour remap** – Plug a palette or reference frame into the *Context* input to nudge the
   final toning toward that colour space.
4. **Video** – Yes it works quite well across video frames! Try experiement with *Initial context for batch* to subtely lock colour across all the processed frames or leave as is for each frame to determine the correction independantly.

### Understanding the *Context* input

The context socket takes a **64 × 64** image that acts as a colour-map reference while the
network predicts its residual.  If you leave it unplugged the node automatically downsamples
your source image – this mirrors how the model was trained and is the safest default.

When might you override it?

* **Palette steer** – pass in a small palette swatch or another artwork to pull the grade towards what would likely be the **correction** 
  those hues and levels.
* **Shot matching** – feed a hero frame from earlier in a sequence to keep later frames corrections more tonally consistent.
* **Creative grading** – drop in a totally different style image (e.g.
a sunset or neon sign) for experimental colour correction shifts.

Remember: the context influences *global* colour balance, not fine detail, so even heavily
down-scaled or abstract images work great but this only plays a small role in the final correction - ultimatley the actual 512x512px patch being corrected has the greated weight in the adjustment task. 

### System Requirements

Works out-of-the-box on CPU via `onnxruntime`.  For **10×** speed improvement ensure ComfyUI can use the `CUDAExecutionProvider` by using onnxruntime-gpu with the hardware that allows this. 
This node print out the currently available providers into the ComfyUI server logs if you need to check this. 


---

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
 
---

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

