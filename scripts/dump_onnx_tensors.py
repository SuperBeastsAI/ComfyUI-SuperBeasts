from pathlib import Path
import csv, onnx

def main():
    onnx_path = Path("models/SuperBeasts_ColorAdjustment_512px_V1.onnx")
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    model = onnx.load(str(onnx_path))
    out_path = Path(__file__).with_name("tensor_blueprint.csv")
    with out_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "name", "shape"])
        for idx, init in enumerate(model.graph.initializer):
            dims = list(init.dims)
            wr.writerow([idx, init.name, dims])
    print(f"[dump] wrote {out_path} with {len(model.graph.initializer)} tensors")

if __name__ == "__main__":
    main() 