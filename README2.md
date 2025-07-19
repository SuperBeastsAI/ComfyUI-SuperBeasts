# SuperBeasts.AI – Custom ComfyUI Nodes & Models


## 🆕 Recent Updates
- **19/07/2025** – *Super Pop Color Adjustment* model + node released.
- **31/07/2024** – Dynamic input bug fixes (thanks @Amorano). Added GitHub Action for Comfy Registry publish (@haohaocreates).
- **30/07/2024** – Deflicker & PixelDeflicker marked Experimental (re-add in workflows if needed).

**Full history:** [CHANGELOG.md](CHANGELOG.md)

---

# 🔥Super Pop Color Adjustment Model & Nodes

<p align="center">
  <img src="examples/SPCA-Header.webp" alt="Super Pop Color Adjustment" width="100%" />
</p>

**Super Pop Color Adjustment** · *Fast residual color-grade engine for vibrant palettes, deep blacks, crisp highlights, HDR bite – in one click.*

---
## 🚀 Quick Start (Super Pop)
1. **Install this repo** into ComfyUI
- Using Comfy Manager search "SuperBeasts.AI"
- OR Clone this repo into `ComfyUI/custom_nodes/`
Clone 
2. Launch ComfyUI.
3. Add **`SB Load Model (SuperBeasts.AI)`** node.
4. Choose model: `SuperPopColorAdjustment/latest` (auto-downloads weights).
5. Connect to **`Super Pop Color Adjustment (SuperBeasts.AI)`** node.
6. Input your image (≤ ~2048px ideal for minimal tile artefacts).
7. Set `Max Strength = 1.0`, `Count = 1`, `Overlap = 0.3–0.4` (higher if >512px).
8. (Optional) Provide *Context* image for palette steer; otherwise leave blank.
9. Execute; adjust Strength / Overlap for taste.
10. Try HDR sandwich (run Super Pop → HDR Effects node) for micro-contrast.

Full usage details: **[Super Pop Docs →](docs/super-pop-color-adjustment.md)**



## 📚 Documentation Index
<table><tr><td valign="top" width="50%">

### Core
- [Super Pop Color Adjustment](docs/super-pop-color-adjustment.md)
- [HDR Effects](docs/hdr-effects.md)
- [Batch Management (Images & Masks)](docs/batch-management.md)
- [Context & Workflow Tips](docs/context-and-workflow.md)

</td><td valign="top" width="50%">

### Reference
- [Deprecated / Removed Nodes](docs/deprecated.md)
- [Contributing](docs/contributing.md)

</td></tr></table>

## ⭐ Support / Community
Follow **[@SuperBeasts.AI](https://www.instagram.com/SuperBeasts.AI)**. 
Feedback / edge cases: open an Issue.


## 🔒 Model & License Snapshot
###Super Color Pop Adjustment Model
Weights: *SPCA-Community-NoSaaS* (local & client use OK, no public SaaS redistribution). Full terms & source: [license-models.md](docs/license-models.md).
