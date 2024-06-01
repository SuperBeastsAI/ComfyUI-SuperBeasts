import { app } from "../../scripts/app.js";

const TypeSlot = {
  Input: 1,
  Output: 2,
};

const TypeSlotEvent = {
  Connect: true,
  Disconnect: false,
};

function node_add_dynamic(nodeType, prefix, type = '*', count = -1) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const me = onNodeCreated?.apply(this);
    this.addInput(`${prefix}1`, type);
    return me;
  };

  const onConnectionsChange = nodeType.prototype.onConnectionsChange;
  nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
    const me = onConnectionsChange?.apply(this, arguments);
    if (slotType === TypeSlot.Input) {
      if (!this.inputs[slot].name.startsWith(prefix)) {
        return;
      }

          // remove all non connected inputs
      if (event == TypeSlotEvent.Disconnect && this.inputs.length > 1) {
        if (this.widgets) {
          const widget = this.widgets.find((w) => w.name === this.inputs[slot].name);
          if (widget) {
            widget.onRemoved?.();
            this.widgets = this.widgets.filter((w) => w.name !== widget.name);
          }
        }
              this.removeInput(slot)

              // make inputs sequential again
        for (let i = 0; i < this.inputs.length; i++) {
          const name = `${prefix}${i + 1}`;
          this.inputs[i].label = name;
          this.inputs[i].name = name;
        }
      }

      if (count - 1 < 0) {
        count = 1000;
      }
      const length = this.inputs.length - 1;
      if (length < count - 1 && this.inputs[length].link != undefined) {
        const nextIndex = this.inputs.length;
        const name = `${prefix}${nextIndex + 1}`;
        this.addInput(name, type);
      }

      if (event === TypeSlotEvent.Connect && link_info) {
        const fromNode = this.graph._nodes.find(
          (otherNode) => otherNode.id == link_info.origin_id
        );
        if (fromNode) {
          const old_type = fromNode.outputs[link_info.origin_slot].type;
          this.inputs[slot].type = old_type;
        }
      } else if (event === TypeSlotEvent.Disconnect) {
        this.inputs[slot].type = type;
        this.inputs[slot].label = `${prefix}${slot + 1}`;
      }
    }
    return me;
  };
  return nodeType;
}

app.registerExtension({
  name: "Comfy.superbeastsai_nodes",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    const dynamicInputConfigs = {
      "Image Batch Manager (SuperBeasts.AI)": { type: "IMAGE", prefix: "image" },
      "Mask Batch Manager (SuperBeasts.AI)": { type: "MASK", prefix: "mask" },
      "String List Manager (SuperBeasts.AI)": { type: "STRING", prefix: "string" },
    };

    if (nodeData.name in dynamicInputConfigs) {
      const { type, prefix } = nodeConfigs[nodeData.name];
      nodeType = node_add_dynamic(nodeType, prefix, type);

      nodeType.prototype.onExecutionStart = function () {

          // Assume 'this.widgets' contains all the widgets of the node
        const max_images = getWidgetValueByName(this.widgets, 'max_images');
        const randomOrderValue = getWidgetValueByName(this.widgets, 'random_order');

        if (randomOrderValue) {
            // Filter image inputs and ensure only valid image slots are considered
          const imageInputs = this.inputs.filter(input => input.type === 'IMAGE');

            // Determine the effective number of indices to consider
          let effectiveLength = Math.min(imageInputs.length, max_images);

            // Ensure we do not accidentally reduce the number of images if not needed
          if (effectiveLength > 0 && imageInputs.length >= max_images) {
              // Generate an array of indices based on the effective length
              let indices = Array.from({length: effectiveLength}, (_, i) => i + 1);
              shuffle(indices); // Shuffle the indices to randomize

              // Convert indices to a string to set as the new value
            const newOrder = indices.join(',');

              // Set the 'new_manual_order' widget value
            setWidgetValueByName(this.widgets, 'new_manual_order', newOrder);
          }
        }
      };
    }
  }
});

// Helper function to shuffle an array (Fisher-Yates shuffle)
function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]]; // Swap elements
  }
}

function getWidgetValueByName(widgets, widgetName) {
    // Check if widgets are defined and iterable
  if (!widgets || !Array.isArray(widgets)) {
    return undefined;
  }

    // Find the widget by name
  const widget = widgets.find(w => w.name === widgetName);
  return widget ? widget.value : undefined;
}

function setWidgetValueByName(widgets, widgetName, value) {
  if (!widgets || !Array.isArray(widgets)) {
    return false;
  }

  const widget = widgets.find(w => w.name === widgetName);
  if (widget) {
    widget.value = value;
    return true;
  }
  return false;
}