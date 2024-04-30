import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.superbeastsai_nodes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name == "Image Batch Manager (SuperBeasts.AI)" ||
        nodeData.name == "Mask Batch Manager (SuperBeasts.AI)") {

            var input_name = "input";

			switch(nodeData.name) {

                case 'Image Batch Manager (SuperBeasts.AI)':
                    input_name = "image";
                    break;    

                case 'Mask Batch Manager (SuperBeasts.AI)':
                    input_name = "mask";
                    break;
			}
            
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (!link_info) return;

                const node = app.graph.getNodeById(link_info.origin_id);
                if (!node) {
                    return;
                }

                let slot_type = node.outputs[link_info.origin_slot].type;
          

                
                let select_slot = this.inputs.find(x => x.name == "select");
				let mode_slot = this.inputs.find(x => x.name == "sel_mode");

				let converted_count = 0;
				converted_count += select_slot?1:0;
				converted_count += mode_slot?1:0;

				if (!connected && (this.inputs.length > 1+converted_count)) {
					const stackTrace = new Error().stack;

					if(
						!stackTrace.includes('LGraphNode.prototype.connect') && // for touch device
						!stackTrace.includes('LGraphNode.connect') && // for mouse device
						!stackTrace.includes('loadGraphData') &&
						this.inputs[index].name != 'select') {
						this.removeInput(index);
					}
				}

				let slot_i = 1;
                
				for (let i = 0; i < this.inputs.length; i++) {
					let input_i = this.inputs[i];
				    if(input_i.name != 'select'&& input_i.name != 'sel_mode') {
						input_i.name = `${input_name}${slot_i}`
						slot_i++;
					}
				}

				let last_slot = this.inputs[this.inputs.length - 1];
                
				if (
					(last_slot.name == 'select' && last_slot.name != 'sel_mode' && this.inputs[this.inputs.length - 2].link != undefined)
					|| (last_slot.name != 'select' && last_slot.name != 'sel_mode' && last_slot.link != undefined)) {
						this.addInput(`${input_name}${slot_i}`, this.outputs[0].type);
				}

            }

        }
    }
});
