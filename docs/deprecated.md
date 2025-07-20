# Deprecated / Removed Nodes
Status reference for legacy or experimental nodes.

## Experimental
| Node | Status | Note |
|------|--------|------|
| Deflicker | Experimental | Not meeting desired deflicker expectations; may be removed in future. |
| PixelDeflicker | Experimental | Same rationale as Deflicker |

> Experimental nodes may disappear without major version bump—avoid hard dependency in critical pipelines.

## Removed
| Node | Replacement | Reason |
|------|-------------|--------|
| CrossFadeImageBatches | Use existing community cross-fade node | Function duplicated elsewhere |
| MakeResizedMaskBatch | MaskBatchManagement | Deprecated please use MaskBatchManagement |

