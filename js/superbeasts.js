import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "Comfy.superbeastsai_nodes",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {

    if (nodeData.name == "Image Batch Manager (SuperBeasts.AI)" || nodeData.name == "Mask Batch Manager (SuperBeasts.AI)") {
		
      var input_type = "IMAGE";
      var input_name = "image";
      switch (nodeData.name) {
        case 'Image Batch Manager (SuperBeasts.AI)':
          input_type = "IMAGE";
          input_name = "image";
          break;
        case 'Mask Batch Manager (SuperBeasts.AI)':
          input_type = "MASK";
          input_name = "mask";
          break;
      }

	  const onConnectionsChange = nodeType.prototype.onConnectionsChange;

	nodeType.prototype.updateSize = function() {
		app.graph.setDirtyCanvas(true);
	};

	  nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
		if (!link_info) return;
	
		const node = app.graph.getNodeById(link_info.origin_id);
		if (!node) {
		  return;
		}
	

		let slot_type;
		if (link_info.origin_slot !== undefined && node.outputs[link_info.origin_slot]) {
		  slot_type = node.outputs[link_info.origin_slot].type;
		}
		let imageInputs = this.inputs.filter(input => input.type === input_type);
		let staticInputs = this.inputs.filter(input => input.type !== input_type);
	
		if (slot_type === input_type) {
		  let empty_slot_count = imageInputs.filter(input => input.link === null).length;
	
		  if (!connected) {
			if (link_info.target_slot !== 0) {
			  let toRemove = empty_slot_count - 1;
			  imageInputs = imageInputs.filter(input => {
				if (input.link === null && toRemove > 0) {
				  toRemove--;
				  return false; // Remove this input
				}
				return true; // Keep this input
			  });
			  
			  // Renumber the remaining image inputs
			  imageInputs.forEach((input, idx) => {
				input.name = `${input_name}${idx + 1}`;
			  });
			}
		  } else {
			// Connection was added
			if (empty_slot_count === 0) {
			  let maxSlotIndex = imageInputs.reduce((max, input) => {
				let matches = input.name.match(/\d+$/);
				return matches ? Math.max(max, parseInt(matches[0], 10)) : max;
			  }, 0);
			  imageInputs.push({ name: `${input_name}${maxSlotIndex + 1}`, type: input_type, link: null });
			}
		  }
		}
	
		// Merge and restore order
		this.inputs = [...staticInputs, ...imageInputs];

		// Update node size after changes
		this.updateSize();
		
	};
	}
  }
});