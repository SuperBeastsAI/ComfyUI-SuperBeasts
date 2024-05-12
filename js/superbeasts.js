import { app } from "../../scripts/app.js";


app.registerExtension({
  name: "Comfy.superbeastsai_nodes",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {

    // Conditionally modify nodes based on their name
    if (nodeData.name == "Image Batch Manager (SuperBeasts.AI)" || nodeData.name == "Mask Batch Manager (SuperBeasts.AI)") {
      let input_type = "IMAGE";
      let input_name = "image";

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

      // Define onNodeCreated method
      nodeType.prototype.onNodeCreated = function() {

        this._type = input_type;
        this.inputs_offset = 0;

        const initialInput = this.addInput(`${input_name}1`, this._type);

      };


	  if (nodeData.name === "Image Batch Manager (SuperBeasts.AI)") {
		nodeType.prototype.onExecutionStart  = function () {
 
			// Assume 'this.widgets' contains all the widgets of the node
			const max_images = getWidgetValueByName(this.widgets, 'max_images');
			const randomOrderValue = getWidgetValueByName(this.widgets, 'random_order');

			if (randomOrderValue) {
				// Filter image inputs and ensure only valid image slots are considered
				const imageInputs = this.inputs.filter(input => input.type === 'IMAGE');

				// Determine the effective number of indices to consider
				let effectiveLength = Math.min(imageInputs.length, max_images);

				// Generate an array of indices based on the effective length
				let indices = Array.from({length: effectiveLength}, (_, i) => i + 1);
				shuffle(indices); // Shuffle the indices to randomize

				// Convert indices to a string to set as the new value
				const newOrder = indices.join(',');

				// Set the 'new_manual_order' widget value
				setWidgetValueByName(this.widgets, 'new_manual_order', newOrder);
			} 

		};	
	  }


      nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
		console.log("Superbeast connection change");

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
              this.removeInput(link_info.target_slot);
            }
          } else {
            // Connection was added
            if (empty_slot_count === 0) {
              let maxSlotIndex = imageInputs.length;
              this.addInput(`${input_name}${maxSlotIndex + 1}`, this._type);
            }
          }
        }

        // Renumber the image inputs
        imageInputs = this.inputs.filter(input => input.type === input_type);
        imageInputs.forEach((input, idx) => {
          input.name = `${input_name}${idx + 1}`;
        });

        // Merge and restore order
        this.inputs = [...staticInputs, ...imageInputs];
      };
    }

	if (nodeData.name == "String List Manager (SuperBeasts.AI)") {
		nodeType.prototype.onNodeCreated = function() {
		  this._type = "STRING";
		  this.inputs_offset = 0;
	  
		  // Add initial input
		  this.addInput(`string1`, this._type);
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
	  
		  let stringInputs = this.inputs.filter(input => input.type === "STRING");
		  let staticInputs = this.inputs.filter(input => input.type !== "STRING");
	  
		  if (slot_type === "STRING") {
			let empty_slot_count = stringInputs.filter(input => input.link === null).length;
			if (!connected) {
			  if (link_info.target_slot !== 0) {
				this.removeInput(link_info.target_slot);
			  }
			} else {
			  // Connection was added
			  if (empty_slot_count === 0) {
				let maxSlotIndex = stringInputs.length;
				this.addInput(`string${maxSlotIndex + 1}`, this._type);
			  }
			}
		  }
	  
		  // Renumber the string inputs
		  stringInputs = this.inputs.filter(input => input.type === "STRING");
		  stringInputs.forEach((input, idx) => {
			input.name = `string${idx + 1}`;
		  });
	  
		  // Merge and restore order
		  this.inputs = [...staticInputs, ...stringInputs];
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
    if (widget) {
        return widget.value;
    } else {
        return undefined;
    }
}


function setWidgetValueByName(widgets, widgetName, value) {
    if (!widgets || !Array.isArray(widgets)) {
        return false;
    }

    const widget = widgets.find(w => w.name === widgetName);
    if (widget) {
        widget.value = value;
        return true;
    } else {
        return false;
    }
}
