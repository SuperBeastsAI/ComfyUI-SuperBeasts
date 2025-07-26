<img src="images/SPCA-Header-opt.webp" width="100%">

# Super Pop Color Adjustment (SuperBeasts.AI)

Bring the **SuperBeasts “pop”** to any image in one click using a new custom residual color correction model trained to push **vibrant palette, deep blacks, crisp highlights, and HDR bite** with ease.

<a href="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/SuperBeasts.AI-SuperPopColorAdjustment.json" target="_blank"><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/workflow.jpg"/></a>
<a href="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/SuperBeasts.AI-SuperPopColorAdjustment.json" target="_blank">Download the ComfyUI Workflow</a>.

### Why I built it
Everything I post gets graded: lift blacks, pull whites, bend curves, punch colour. Existing auto tools (and many AI outputs — looking at you, muddy yellow “whites”) just weren’t landing, especially across batches or iterative generations. Training my own lightweight correction model let me reclaim hours of post and lock in a consistent SuperBeasts look I could reuse — and share.

**Personal project, shared as‑is.**  
Works great generally across my SuperBeasts production flow and a wide mix of test art, but it’s *not* exhaustively validated on every style, lighting condition, or colour space. Expect edge cases, occasional under or overcorrection, and behaviour that varies with resolution/patch overlap. But by and large it's still a huge timer saver in my opinion so please experiment and let me know what you think.

If you want to show support please take a momenet to follow me on Instagram <a href="https://www.instagram.com/SuperBeasts.AI" target="_blank">@SuperBeasts.AI</a>

Happy generating!!

---

### Using Super Pop

Before starting please review the <a href="https://github.com/SuperBeastsAI/SuperBeastsAI-Models/blob/main/SuperPopColorAdjustment/LICENSE.txt" target="_blank">the licence</a> and [System Requirements](#System-Requirements) to run onnx models.

1. Add **SB Load Model (SuperBeasts.AI)** node.
2. Pick **SuperPopColorAdjustment/latest**. (Now from V2 using .safetensors)
3a. Connect to **Super Pop Color Adjustment (SuperBeasts.AI)** node.
3b. Input source image and tune *Max Strength*, *Count*, *Overlap* (recommendations below)
4. Optional - Use the **Super Pop Residual Blend (SuperBeasts.AI)** for full control of the strength. (Useful to experiement with strength settings without repeat runs of the model)

**Auto model weight download:**
Weights are automatically downloaded the first time the **SB Load Model** node runs using a specific version or 'latest'.

If you prefer manual download, grab them from the GitHub release page:
<a href="https://github.com/SuperBeastsAI/SuperBeastsAI-Models/tree/main/SuperPopColorAdjustment/" target="_blank">https://github.com/SuperBeastsAI/SuperBeastsAI-Models/tree/main/SuperPopColorAdjustment/</a>
and drop the `.onnx` file into `custom_nodes/ComfyUI-SuperBeasts/models/`.

**License:**
Weights are SPCA-Community-NoSaaS. 
Local/client use OK; no public SaaS redistribution.
<a href="https://github.com/SuperBeastsAI/SuperBeastsAI-Models/blob/main/SuperPopColorAdjustment/LICENSE.txt" target="_blank">See licence in the model repo</a>.


<h2>Same settings multiple corrections (Stength: 1.0)</h2>

<h3>Highlights/Shadows Correction</h3>

<table width="100%">
  <tr>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
    <td style="width:25%" width="25%">Original</td>
    <td style="width:25%" width="25%">SPCA</td>
  </tr>
  <tr>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/5a-1024px.png" width="100%" alt="Original 1"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/5b-1024px.png" width="100%" alt="SPCA 1"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/1a-1024px.png" width="100%" alt="Original 2"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/1b-1024px.png" width="100%" alt="SPCA 2"></td>
  </tr>
  <tr>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/2a-1024px.png" width="100%" alt="Original 3"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/2b-1024px.png" width="100%" alt="SPCA 3"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/4a-1024px.png" width="100%" alt="Original 4"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/4b-1024px.png" width="100%" alt="SPCA 4"></td>
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
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/41a-1024px.png" width="100%" alt="Original 5"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/41b-1024px.png?id-1" width="100%" alt="SPCA 5"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/16a-1024px.png" width="100%" alt="Original 6"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/16b-1024px.png" width="100%" alt="SPCA 6"></td>
  </tr>
  <tr>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/27a-1024px.png" width="100%" alt="Original 7"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/27b-1024px.png" width="100%" alt="SPCA 7"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/24a-1024px.png" width="100%" alt="Original 8"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/24b-1024px.png" width="100%" alt="SPCA 8"></td>
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
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/6a-1024px.png" width="100%" alt="Original 9"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/6b-1024px.png" width="100%" alt="SPCA 9"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/9a-1024px.png?id=1" width="100%" alt="Original 10"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/9b-1024px.png" width="100%" alt="SPCA 10"></td>
  </tr>
  <tr>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/38a-1024px.png" width="100%" alt="Original 11"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/38b-1024px.png" width="100%" alt="SPCA 11"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/26a-1024px.png" width="100%" alt="Original 12"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/26b-1024px.png" width="100%" alt="SPCA 12"></td>
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
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/7a-1024px.png" width="100%" alt="Original 13"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/7b-1024px.png" width="100%" alt="SPCA 13"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/14a-1024px.png" width="100%" alt="Original 14"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/14b-1024px.png" width="100%" alt="SPCA 14"></td>
  </tr>
  <tr>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/13a-1024px.png" width="100%" alt="Original 15"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/13b-1024px.png" width="100%" alt="SPCA 15"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/34a-1024px.png" width="100%" alt="Original 16"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/34b-1024px.png" width="100%" alt="SPCA 16"></td>
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
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/25a-1024px.png" width="100%" alt="Original 17"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/25b-1024px.png?id=1" width="100%" alt="SPCA 17"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/21a-1024px.png" width="100%" alt="Original 18"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/21b-1024px.png" width="100%" alt="SPCA 18"></td>
  </tr>
  <tr>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/35a-1024px.png" width="100%" alt="Original 19"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/35b-1024px.png" width="100%" alt="SPCA 19"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/28a-1024px.png" width="100%" alt="Original 20"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/28b-1024px.png?id=2" width="100%" alt="SPCA 20"></td>
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
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/18a-1024px.png" width="100%" alt="Original 21"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/18b-1024px.png" width="100%" alt="SPCA 21"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/39a-1024px.png" width="100%" alt="Original 22"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/39b-1024px.png" width="100%" alt="SPCA 22"></td>
  </tr>
  <tr>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/37a-1024px.png?id=1" width="100%" alt="Original 23"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/37b-1024px.png" width="100%" alt="SPCA 23"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/40a-1024px.png" width="100%" alt="Original 24"></td>
    <td><img src="https://s3.ap-southeast-2.amazonaws.com/superbeasts.ai/repos/ComfyUI-SuperBeasts/images/samples/40b-1024px.png" width="100%" alt="SPCA 24"></td>
  </tr>
</table>


### Recommended quick-start

| Parameter | Suggested | Notes |
|-----------|-----------|-------|
| Image | Image dimensions ideally <2048px | Techncially supports any size but due to 512px patch size local adjustments become increasingly obvious as size increases. |
| Max strength  | **1.0**   | 0.5-2.0 for subtle ➜ dramatic |
| Count     | 1         | Increase for automatic strength batching up to your max strength value. E.g. Max strength 1.0 with count 2.0 runs strength at 0.5 and 1.0 outputting 2 images |
| Overlap   | At least 0.3-0.4 if image is >512px  | Ideally up to 0.9 if GPU / Compute permits for maximum quality. |
| Context (Image) | Leave empty | See detailed notes on context below. |
| Initial context for batch | False | Use the first input images context for all images processed. If set to True this may be useful to reduce correction variation across the batch E.g. Video frames |

### How it works

1. **SB Load Model** node loads the ONNX-formatted colour-adjustment network
    (`models/SuperBeasts_ColorAdjustment_512px_V1.onnx`).
2. The model analyses the image in 512 × 512 patches, predicting a residual colour grade.
3. Patches are seamlessly stitched back with configurable **strength**, **count** (variant
    batching) and **overlap** for large-resolution output.
4. A 64 × 64 **context** thumbnail guides the global colour mapping – leave it empty for
    natural results or experiment with creative transfers.

### Known limitations / quirks

* **Early and personalised model:** Really this was a model built for personal use for my particular flavour for colour/levels adjustment. It's intentionally meant to be simple and have limited settings so it's not going work for everyone. Future training would be needed to improve it's range or configuration options (Not planned at this stage). 
* **Patch size sensitivity:** The network works on 512 px tiles – with images above ~2 K you may
  notice slight local exposure shifts.  Pushing *Overlap* up to 0.8-0.9 blends these out much better for higher quality results at the
  cost of extra compute/time.
* **Atmospheric overlays:** Because the model corrects blacks / whites aggressively it can
  dial back matte-style colour washes or cinematic tints.  Keep *Strength* around 0.7, or increase count for variation options or blend
  the result back with the original.
* **Very high strengths:** Values above 2.0 can clip highlights or oversaturate; use sparingly
  for stylised effects.

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


### Creative workflow ideas

1. **HDR Sandwich** – Run Super Pop, then feed the output into **HDR Effects** (with gentle
   settings) to amplify micro-contrast.
2. **Colour remap** – Plug a palette or reference frame into the *Context* input to nudge the
   final toning into what would essentially be a correction for that provided colour space.
3. **Video** – It works quite well across video frames! You could try experiement with *Initial context for batch* to subtely lock colour correction guidance across all the processed frames or leave as is for each frame to determine the correction independantly.

### System Requirements

#### Latest version 2.0 uses .safetensors

From version 2.0 .safetensors model is in use with corresponding .pt file this model now runs natively in standard Comfy environment. 
GPU is still recommended where you will acheive speeds such as: 

- 1024x1024px with 0.5 overlap = 1.63s
- 2048x2048px with 0.5 overlap = 5.23s

#### Legacy 1.0 Onnx Model

Works out-of-the-box on CPU via `onnxruntime` but it is extremely slow and I wouldn't recommend it. 
For **10×** speed improvement ensure ComfyUI can use the `CUDAExecutionProvider` by installing `onnxruntime-gpu` into the ComfyUI environment. 

This node currently prints out the available providers for onnxruntime into the ComfyUI server logs when you run the node. So if you are not seeing `CUDAExecutionProvider` it's likely the model is using CPU instead. 

For reference I use:
Python version: 3.12.1 
pytorch version: 2.5.0+cu124
NVIDIA GeForce RTX 4090 
with onnxruntime-gpu installed in the ComfyUI Environment
and C:\Program Files\NVIDIA\CUDNN\vX.X\bin on your System Path (where x.x is your version)
For more on debugging try tips from the user here: https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts/issues/10#issuecomment-3113530795

Server Logs show: 
[SuperBeasts] ONNX Runtime providers available: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
[SuperBeasts] Using ORT providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']

