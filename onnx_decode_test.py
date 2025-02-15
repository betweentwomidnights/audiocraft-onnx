#!/usr/bin/env python3
import torch
import torch.nn as nn
from audiocraft.models.encodec import CompressionModel

class CompressionDecodeWrapper(nn.Module):
    def __init__(self, compression_model):
        super().__init__()
        self.compression_model = compression_model

    def forward(self, codes: torch.Tensor):
        # codes: shape [B, K, T]
        # scale is optional; if your model uses scale, add it as a second input
        return self.compression_model.decode(codes, scale=None)

def main():
    model_path = "compression_state_dict.bin"
    output_file = "compression_decode.onnx"

    device = torch.device("cpu")
    model = CompressionModel.get_pretrained(model_path, device=device).eval()

    # Create the decode-only wrapper
    wrapper = CompressionDecodeWrapper(model).eval()

    # Example: K=4 codebooks, T=100 tokens
    dummy_codes = torch.randint(0, model.cardinality, (1, model.num_codebooks, 100))
    dynamic_axes = {
        "codes": {2: "T_codes"},
        "output": {2: "T_audio"}
    }

    torch.onnx.export(
        wrapper,
        dummy_codes,
        output_file,
        opset_version=14,
        input_names=["codes"],
        output_names=["decoded_audio"],
        dynamic_axes=dynamic_axes
    )
    print(f"Decoder ONNX exported to {output_file}")

if __name__ == "__main__":
    main()
