#!/usr/bin/env python3
import torch
import numpy as np
import onnxruntime as ort
from audiocraft.models.lm import LMModel
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from audiocraft.modules.conditioners import ConditioningProvider, ConditionFuser
import json
from typing import Dict, Optional, Tuple, List

class MinimalConditioningProvider(ConditioningProvider):
    def __init__(self, device="cpu"):
        super().__init__({}, device=device)
        
    def tokenize(self, conditions: List) -> Dict:
        return {}

    def forward(self, tokenized: Dict) -> Dict:
        return {}

class MinimalConditionFuser(ConditionFuser):
    def __init__(self):
        super().__init__(
            fuse2cond={
                "prepend": [],
                "cross": [],
                "sum": [],
                "input_interpolate": []
            }
        )

def load_pytorch_model(checkpoint_path: str, device: str = "cpu") -> LMModel:
    """Load PyTorch model with minimal conditioning."""
    pattern_provider = DelayedPatternProvider(
        n_q=4,
        delays=[0, 1, 2, 3],
        flatten_first=0,
        empty_initial=0
    )
    
    condition_provider = MinimalConditioningProvider(device)
    fuser = MinimalConditionFuser()
    
    model = LMModel(
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
        bias_attn=False
    )
    
    state = torch.load(checkpoint_path, map_location=device)
    if "best_state" in state:
        state = state["best_state"]
        
    filtered_state = {}
    for k, v in state.items():
        # Skip condition provider and cross attention components
        if k.startswith('condition_provider') or 'cross_attention' in k or 'norm_cross' in k:
            continue
            
        # Convert attention layer naming
        if 'self_attn.in_proj_weight' in k:
            new_k = k.replace('self_attn.in_proj_weight', 'self_attn.mha.in_proj_weight')
            filtered_state[new_k] = v
        elif 'self_attn.out_proj.weight' in k:
            new_k = k.replace('self_attn.out_proj.weight', 'self_attn.mha.out_proj.weight')
            filtered_state[new_k] = v
        else:
            filtered_state[k] = v
    
    # Debug: Print what keys we're loading
    print("\nLoading keys:")
    for k in sorted(filtered_state.keys()):
        print(f"  {k}")
        
    print("\nModel's state_dict keys:")
    for k in sorted(model.state_dict().keys()):
        print(f"  {k}")
    
    model.load_state_dict(filtered_state, strict=True)
    model.eval()
    return model

def analyze_sequence_step(
    step: int,
    sequence: torch.Tensor,
    pattern_provider: DelayedPatternProvider,
    special_token_id: int
) -> Dict:
    """Analyze a single step of sequence construction."""
    pattern = pattern_provider.get_pattern(sequence.shape[-1])
    sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
        sequence, special_token_id, keep_only_valid_steps=True
    )
    
    return {
        'step': step,
        'sequence_shape': tuple(sequence.shape),
        'sequence_codes_shape': tuple(sequence_codes.shape),
        'valid_positions': sequence_mask.sum().item(),
        'special_tokens': (sequence_codes == special_token_id).sum().item(),
        'last_step_mask': sequence_mask[..., -1].tolist(),
        'sequence_values': sequence.tolist(),
        'sequence_codes_values': sequence_codes.tolist()
    }

def analyze_logits(logits: torch.Tensor, top_k: int = 5) -> Dict:
    """Analyze logits distribution and top-k values.
    
    Args:
        logits: Tensor of shape [B, K, card] or [B, K, T, card]
        top_k: Number of top values to return
    Returns:
        Dictionary containing distribution statistics and top-k values
    """
    # Ensure tensor is contiguous and on CPU for analysis
    logits = logits.contiguous()
    if logits.device != torch.device('cpu'):
        logits = logits.cpu()
    
    # Flatten logits for top-k while preserving batch and codebook dimensions
    B, K = logits.shape[:2]
    if logits.dim() == 4:  # [B, K, T, card]
        logits = logits[..., -1]  # Take last timestep
    
    # Now logits should be [B, K, card]
    flattened = logits.reshape(-1)  # Combine all dimensions for analysis
    
    # Get distribution statistics
    stats = {
        'shape': tuple(logits.shape),
        'mean': float(logits.mean().item()),
        'std': float(logits.std().item()),
        'min': float(logits.min().item()),
        'max': float(logits.max().item()),
        'has_nan': bool(torch.isnan(logits).any().item()),
        'percentiles': {
            '0': float(torch.quantile(flattened, 0).item()),
            '25': float(torch.quantile(flattened, 0.25).item()),
            '50': float(torch.quantile(flattened, 0.50).item()),
            '75': float(torch.quantile(flattened, 0.75).item()),
            '100': float(torch.quantile(flattened, 1.0).item()),
        }
    }
    
    # Get top-k values
    topk_values, topk_indices = torch.topk(flattened, min(top_k, flattened.numel()))
    stats['top_k'] = [
        (int(idx.item()), float(val.item()))
        for idx, val in zip(topk_indices, topk_values)
    ]
    
    return stats

def debug_pytorch_step(
    model: LMModel,
    sequence: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 250
) -> Dict:
    """Debug a single step of PyTorch model generation."""
    with torch.no_grad():
        # Get pattern sequence info
        sequence_info = analyze_sequence_step(
            0, sequence, model.pattern_provider, model.special_token_id
        )
        
        # Get logits
        logits = model(
            sequence,
            conditions=[],
            condition_tensors={},
            stage=-1
        )
        
        # Permute logits to [B, K, card, T] and take last timestep
        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        last_step_logits = logits[..., -1]  # [B, K, card]
        logits_analysis = analyze_logits(last_step_logits)
        
    return {
        'sequence_info': sequence_info,
        'logits_analysis': logits_analysis
    }

def debug_onnx_step(
    session: ort.InferenceSession,
    sequence: np.ndarray,
    special_token_id: int,
    pattern_provider: DelayedPatternProvider,
    temperature: float = 1.0,
    top_k: int = 250
) -> Dict:
    """Debug a single step of ONNX model generation."""
    # Convert sequence to torch for pattern analysis
    sequence_torch = torch.from_numpy(sequence)
    sequence_info = analyze_sequence_step(
        0, sequence_torch, pattern_provider, special_token_id
    )
    
    # Get ONNX logits and convert to torch tensor
    logits = session.run(None, {"sequence": sequence.astype(np.int64)})[0]
    logits_torch = torch.from_numpy(logits).contiguous()
    logits_analysis = analyze_logits(logits_torch)
    
    return {
        'sequence_info': sequence_info,
        'logits_analysis': logits_analysis
    }

def compare_generation_steps(
    pytorch_model: LMModel,
    onnx_session: ort.InferenceSession,
    num_steps: int = 5,
    sequence_length: int = 10,
    save_path: str = "debug_output.json"
):
    """Compare PyTorch and ONNX generation steps."""
    device = "cpu"
    B = 1
    K = pytorch_model.num_codebooks
    
    # Create same random sequence for both
    torch.manual_seed(42)
    sequence = torch.randint(0, pytorch_model.card, (B, K, sequence_length), device=device)
    sequence_np = sequence.numpy()
    
    results = {
        'pytorch': [],
        'onnx': [],
        'comparison': []
    }
    
    for step in range(num_steps):
        print(f"\nAnalyzing step {step}...")
        
        # Get PyTorch and ONNX debug info
        pytorch_debug = debug_pytorch_step(pytorch_model, sequence)
        onnx_debug = debug_onnx_step(
            onnx_session, sequence_np, 
            pytorch_model.special_token_id,
            pytorch_model.pattern_provider
        )
        
        # Add to results
        results['pytorch'].append(pytorch_debug)
        results['onnx'].append(onnx_debug)
        
        # Compare key metrics
        step_comparison = {
            'step': step,
            'sequence_shape_match': pytorch_debug['sequence_info']['sequence_shape'] == onnx_debug['sequence_info']['sequence_shape'],
            'valid_positions_match': pytorch_debug['sequence_info']['valid_positions'] == onnx_debug['sequence_info']['valid_positions'],
            'special_tokens_match': pytorch_debug['sequence_info']['special_tokens'] == onnx_debug['sequence_info']['special_tokens'],
            'logits_shape_match': pytorch_debug['logits_analysis']['shape'] == onnx_debug['logits_analysis']['shape'],
            'max_logit_diff': abs(pytorch_debug['logits_analysis']['max'] - onnx_debug['logits_analysis']['max']),
            'mean_logit_diff': abs(pytorch_debug['logits_analysis']['mean'] - onnx_debug['logits_analysis']['mean']),
            'pytorch_has_nan': pytorch_debug['logits_analysis']['has_nan'],
            'onnx_has_nan': onnx_debug['logits_analysis']['has_nan'],
            'top_k_overlap': len(
                set(k for k,v in pytorch_debug['logits_analysis']['top_k']) &
                set(k for k,v in onnx_debug['logits_analysis']['top_k'])
            )
        }
        results['comparison'].append(step_comparison)
        
        # Print key differences
        print(f"  Sequence shapes match: {step_comparison['sequence_shape_match']}")
        print(f"  Valid positions match: {step_comparison['valid_positions_match']}")
        print(f"  Special tokens match: {step_comparison['special_tokens_match']}")
        print(f"  Logits shape match: {step_comparison['logits_shape_match']}")
        print(f"  Max logit difference: {step_comparison['max_logit_diff']:.6f}")
        print(f"  Mean logit difference: {step_comparison['mean_logit_diff']:.6f}")
        print(f"  PyTorch has NaN: {step_comparison['pytorch_has_nan']}")
        print(f"  ONNX has NaN: {step_comparison['onnx_has_nan']}")
        print(f"  Top-k token overlap: {step_comparison['top_k_overlap']}/5")
        
        # Extend sequence with random token for next step
        sequence = torch.cat([
            sequence,
            torch.randint(0, pytorch_model.card, (B, K, 1), device=device)
        ], dim=-1)
        sequence_np = sequence.numpy()
    
    # Save detailed results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed analysis to {save_path}")

def main():
    # Load models
    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model("state_dict.bin")
    
    print("Loading ONNX model...")
    onnx_session = ort.InferenceSession("lm_inference.onnx")
    
    # Run comparison
    compare_generation_steps(
        pytorch_model=pytorch_model,
        onnx_session=onnx_session,
        num_steps=5,  # Number of generation steps to analyze
        sequence_length=10,  # Initial sequence length
        save_path="debug_output.json"
    )

if __name__ == "__main__":
    main()