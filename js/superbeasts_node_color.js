// Register a small ComfyUI frontend extension that forces a
// consistent colour for every SuperBeasts.AI custom node as soon as it
// appears (either when a workflow is loaded or when the user drags a
// node from the sidebar).
//
// The extension is intentionally self-contained so it can be dropped
// into `custom_nodes/ComfyUI-SuperBeasts/js/` without requiring any
// further changes on the Python side.

(() => {
  const SUPERBEASTS_HEX = "#000048"; // SuperBeastsBlue

  /**
   * Returns true if the node belongs to the SuperBeasts.AI pack.
   * We use both the `type` (shown in the sidebar) and the internal
   * `comfyClass` string to be future-proof against renames.
   */
  function isSuperBeastsNode(node) {
    if (!node) return false;
    const t = (node.type || "") + " " + (node.comfyClass || "");
    return /superbeasts/i.test(t);
  }

  /** Apply the colour to a node in-place */
  function paint(node) {
    if (!isSuperBeastsNode(node)) return;
    // Respect any colour the user (or a saved workflow) already set.
    // Only apply our default if the node has no explicit colour yet.
    if (node.bgcolor === undefined || node.bgcolor === null) {
      node.bgcolor = SUPERBEASTS_HEX;
    }
    if (node.color === undefined || node.color === null) {
      node.color = SUPERBEASTS_HEX;
    }
  }

  app.registerExtension({
    name: "superbeasts.defaultNodeColour",
    init() {
      // Colour newly created nodes (when user drags from the sidebar).
      app.on("nodeCreated", (node) => {
        paint(node);
        // refresh canvas to show the new colour immediately
        app.graph.setDirtyCanvas(true, true);
      });
    },
  });
})(); 