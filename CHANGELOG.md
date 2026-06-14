# Changelog
All notable changes to this project will be documented here. Dates in **DD/MM/YYYY**.

## 14/06/2026
### Fixed
- **HDR Effects** – replaced Pillow `ImageCms` LAB conversion with a pure NumPy path to stop hard segfaults on large frames (thanks [@xmarre](https://github.com/xmarre), [#14](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts/pull/14)).
- Hardened model cache and ONNX Runtime threading to reduce prompt-worker freezes under repeated runs.
- Reduced peak memory for HDR batch processing and Super Pop Color Adjustment blending.

## 19/07/2025
### Added
- Released *Super Pop Color Adjustment* node + ONNX model (automatic weight fetch).

## 31/07/2024
### Fixed / Added
- Dynamic input bugs resolved (thanks @Amorano).
- GitHub Action for publishing to Comfy Registry (@haohaocreates).

## 30/07/2024
### Changed
- Marked *Deflicker* & *PixelDeflicker* as Experimental labels.

## 30/04/2024
### Added
- Deflicker and PixelDeflicker nodes (reduce flicker in sequences) – now Experimental.
- CrossFadeImageBatches node (since removed – see deprecated).
- Batch Management nodes improvements: resizing, cropping, reordering.

## 27/03/2024
### Added
- Initial public release; update resolving batched images for videos.