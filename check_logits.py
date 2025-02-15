#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import onnxruntime as ort
from audiocraft.models.lm import LMModel
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from audiocraft.modules.conditioners import ConditioningProvider, ConditionFuser

class DebugLMModel(LMModel):
    """Wrapper around LMModel to capture layer outputs"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_outputs = []
        
    def forward(self, *args, **kwargs):
        self.layer_outputs = []
        
        # Hook to capture layer outputs
        def hook_fn(module, input, output):
            self.layer_outputs.append(output.detach().cpu())
            
        hooks = []
        for layer in self.transformer.layers:
            hooks.append(layer.register_forward_hook(hook_fn))
            
        try:
            output = super().forward(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
                
        return output

def load_pytorch_lm_model(checkpoint_path: str, device: torch.device) -> LMModel:
    pattern_provider = DelayedPatternProvider(
        n_q=4,
        delays=[0, 1, 2, 3],
        flatten_first=0,
        empty_initial=0
    )
    condition_provider = ConditioningProvider([])
    fuser = ConditionFuser(
        fuse2cond={
            "prepend": [],
            "cross": [],
            "sum": [],
            "input_interpolate": []
        }
    )
    
    # Use DebugLMModel instead of LMModel
    lm = DebugLMModel(
        pattern_provider=pattern_provider,
        condition_provider=condition_provider,
        fuser=fuser,
        n_q=4,
        card=2048,
        dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_scale=4,
        norm='layer_norm',
        norm_first=True,
        emb_lr=None,
        bias_proj=False,
        bias_ff=False,
        bias_attn=False,
        weight_init='gaussian',
        depthwise_init='current',
        zero_bias_init=True,
        attribute_dropout={},
        two_step_cfg=False
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if "best_state" in state:
        state = state["best_state"]
    lm.load_state_dict(state, strict=False)
    lm.eval()
    return lm

def analyze_layer_outputs(pytorch_model, onnx_outputs):
    """Analyze intermediate layer outputs"""
    print("\nLayer Analysis:")
    for i, layer_output in enumerate(pytorch_model.layer_outputs):
        stats = {
            'mean': layer_output.mean().item(),
            'std': layer_output.std().item(),
            'min': layer_output.min().item(),
            'max': layer_output.max().item()
        }
        print(f"\nLayer {i}:")
        print(f"  Shape: {layer_output.shape}")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Min:  {stats['min']:.3f}")
        print(f"  Max:  {stats['max']:.3f}")

def validate_masks(pytorch_model, sequence):
    """Validate masking between PyTorch and pattern sequence building"""
    pattern = pytorch_model.pattern_provider.get_pattern(sequence.shape[-1])
    sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
        sequence,
        pytorch_model.special_token_id,
        keep_only_valid_steps=False
    )
    
    print("\nMask Validation:")
    print(f"Sequence shape: {sequence_codes.shape}")
    print(f"Mask shape: {sequence_mask.shape}")
    print(f"Valid positions in mask: {sequence_mask.sum().item()}")
    print(f"Special tokens: {(sequence_codes == pytorch_model.special_token_id).sum().item()}")
    
    # Verify last timestep specifically
    last_step_mask = sequence_mask[..., -1]
    print("\nLast timestep analysis:")
    print(f"Valid positions in last step: {last_step_mask.sum().item()}")
    return sequence_codes, sequence_indexes, sequence_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx_model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt_length", type=int, default=5)
    parser.add_argument("--codebooks", type=int, default=4)
    parser.add_argument("--cardinality", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading PyTorch LM from: {args.checkpoint}")
    pytorch_lm = load_pytorch_lm_model(args.checkpoint, device)

    B, K, T = 1, args.codebooks, args.prompt_length
    prompt_tokens_torch = torch.randint(0, args.cardinality, (B, K, T), dtype=torch.long, device=device)
    print(f"Random prompt shape: {prompt_tokens_torch.shape}")

    with torch.no_grad():
        out = pytorch_lm.compute_predictions(
            codes=prompt_tokens_torch,
            conditions=[],
            condition_tensors={},
            keep_only_valid_steps=False  # Match ONNX wrapper
        )
        pytorch_logits = out.logits[:, :, -1, :]
        
    # Analyze layer outputs
    analyze_layer_outputs(pytorch_lm, None)
    
    pytorch_logits_np = pytorch_logits.cpu().numpy()

    print(f"\nLoading ONNX model from: {args.onnx_model}")
    onnx_sess = ort.InferenceSession(args.onnx_model)

    prompt_np = prompt_tokens_torch.cpu().numpy()
    onnx_outputs = onnx_sess.run(None, {"sequence": prompt_np})
    onnx_logits = onnx_outputs[0]

    print(f"\nPyTorch logits shape: {pytorch_logits_np.shape}")
    print(f"ONNX logits shape:    {onnx_logits.shape}")

    diff = pytorch_logits_np - onnx_logits
    max_abs_diff = np.abs(diff).max()
    mean_abs_diff = np.abs(diff).mean()

    print(f"\nMax abs difference between PyTorch and ONNX last-step logits: {max_abs_diff:.6f}")
    print(f"Mean abs difference: {mean_abs_diff:.6f}")

    # Print distributions
    k_example = 0
    topn = 5
    
    pt_slice = pytorch_logits_np[0, k_example]
    onnx_slice = onnx_logits[0, k_example]
    
    print(f"\nLogits distribution for codebook {k_example}:")
    percentiles = [0, 25, 50, 75, 100]
    print("PyTorch percentiles:")
    for p, v in zip(percentiles, np.percentile(pt_slice, percentiles)):
        print(f"  {p}th: {v:.3f}")
    print("ONNX percentiles:")
    for p, v in zip(percentiles, np.percentile(onnx_slice, percentiles)):
        print(f"  {p}th: {v:.3f}")

    pt_sort_idx = np.argsort(-pt_slice)
    onnx_sort_idx = np.argsort(-onnx_slice)

    sequence_codes, sequence_indexes, sequence_mask = validate_masks(pytorch_lm, prompt_tokens_torch)

    print(f"\nPyTorch top-{topn} tokens:")
    for i in range(topn):
        tok = pt_sort_idx[i]
        print(f"Token {tok}, logit={pt_slice[tok]:.3f}")

    print(f"\nONNX top-{topn} tokens:")
    for i in range(topn):
        tok = onnx_sort_idx[i]
        print(f"Token {tok}, logit={onnx_slice[tok]:.3f}")

if __name__ == "__main__":
    main()