#!/usr/bin/env python3
import torch
import torch.nn as nn
from audiocraft.models.lm import LMModel
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from audiocraft.modules.conditioners import ConditioningProvider, ConditionFuser
from audiocraft.modules.transformer import StreamingTransformer, create_norm_fn
from typing import Dict, Optional, Tuple, List

class MinimalConditioner(nn.Module):
    """A minimal conditioner with matching structure to checkpoint."""
    def __init__(self, dim_in: int = 768, dim_out: int = 1024):
        super().__init__()
        self.output_proj = nn.Linear(dim_in, dim_out)

class MinimalConditioningProvider(ConditioningProvider):
    """Minimal but functional conditioning provider that maintains required interface."""
    def __init__(self, device="cpu"):
        super().__init__({}, device=device)
        self.conditioners = nn.ModuleDict({
            'description': MinimalConditioner()
        })
    
    def tokenize(self, conditions: List) -> Dict:
        return {}

    def forward(self, tokenized: Dict) -> Dict:
        return {}

class MinimalConditionFuser(ConditionFuser):
    """Minimal but functional fuser that maintains expected interface."""
    def __init__(self):
        super().__init__(
            fuse2cond={
                "prepend": [],
                "cross": [],
                "sum": [],
                "input_interpolate": []
            }
        )

    def forward(self, x: torch.Tensor, conditions: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return x, None

class TracerFriendlyPattern:
    """ONNX-friendly implementation of pattern building logic using only PyTorch operations."""
    
    def __init__(self, n_q: int, delays: List[int]):
        self.n_q = n_q
        self.delays = delays
        
    def build_sequence_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build sequence mask in a tracer-friendly way."""
        # Create position indices
        t = torch.arange(T, device=device)
        q = torch.arange(self.n_q, device=device)
        delays = torch.tensor(self.delays, device=device)
        
        # Expand for broadcasting
        t = t.view(1, 1, T)  # [1, 1, T]
        q = q.view(1, self.n_q, 1)  # [1, K, 1]
        delays = delays.view(1, self.n_q, 1)  # [1, K, 1]
        
        # Compute valid positions with broadcasting
        valid_pos = t >= delays  # [1, K, T]
        return valid_pos

class LMInferenceWrapper(nn.Module):
    """ONNX-exportable wrapper that implements single-step prediction with explicit scaling debug."""
    
    def __init__(self, lm_model: LMModel):
        super().__init__()
        self.lm_model = lm_model
        self.pattern_provider = lm_model.pattern_provider
        self.special_token_id = lm_model.special_token_id
        self.card = lm_model.card
        self.num_heads = lm_model.transformer.layers[0].self_attn.num_heads
        self.dim = lm_model.dim
        self.head_dim = self.dim // self.num_heads
        # Register a large negative value instead of -inf for masking
        self.register_buffer('neg_value', torch.tensor(-1e4, dtype=torch.float32))
        # Create tracer-friendly pattern builder
        self.tracer_pattern = TracerFriendlyPattern(
            n_q=lm_model.pattern_provider.n_q,
            delays=lm_model.pattern_provider.delays
        )
        
    def _debug_tensor(self, name: str, tensor: torch.Tensor):
        """Debug helper that won't be traced."""
        if not hasattr(self, 'debug_prints') or not self.debug_prints:
            return
        
        print(f"\n{name}:")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        
        if tensor.dtype == torch.bool:
            print(f"Values: {tensor.float().mean().item():.3f} (mean)")
        elif tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            print(f"Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
            print(f"Mean: {tensor.mean().item():.3f}")
            print(f"Std: {tensor.std().item():.3f}")
        else:  # Integer types
            print(f"Range: [{tensor.min().item()}, {tensor.max().item()}]")
            print(f"Mean: {tensor.float().mean().item():.3f}")
            print(f"Std: {tensor.float().std().item():.3f}")
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Single step prediction with explicit scaling debug."""
        B, K, T = sequence.shape
        
        # Get sequence mask using tracer-friendly implementation
        sequence_mask = self.tracer_pattern.build_sequence_mask(T, sequence.device)
        self._debug_tensor("Sequence mask", sequence_mask)
        
        # Get pattern sequence (matches logic in compute_predictions)
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, pattern_mask = pattern.build_pattern_sequence(
            sequence, 
            self.special_token_id,
            keep_only_valid_steps=False  # Match original model behavior
        )
        self._debug_tensor("Sequence codes", sequence_codes)
        
        # Forward through model
        logits = self.lm_model(
            sequence_codes,
            conditions=[],
            condition_tensors={},
            stage=-1
        )  # [B, S, card]
        self._debug_tensor("Initial logits", logits)
        
        # Process logits following compute_predictions logic
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # Revert pattern logits like in compute_predictions
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, self.neg_value.item(), keep_only_valid_steps=False
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        self._debug_tensor("Reshaped logits", logits)
        
        # Extract last timestep logits
        logits = logits[..., -1, :]  # [B, K, card]
        self._debug_tensor("Final logits", logits)
        
        # Get the mask for the last timestep from pattern_mask
        # pattern_mask is [K, S], we want the last valid position for each K
        last_valid_pos = pattern_mask.sum(dim=1) - 1  # [K]
        last_step_mask = torch.zeros_like(pattern_mask)  # [K, S]
        for k in range(K):
            if last_valid_pos[k] >= 0:  # Check if there are any valid positions
                last_step_mask[k, last_valid_pos[k]] = True
        
        # Reshape mask for broadcasting with logits
        last_step_mask = last_step_mask.any(dim=1)  # [K]
        last_step_mask = last_step_mask.view(1, K, 1).expand(B, -1, self.card)
        self._debug_tensor("Last step mask", last_step_mask)
        
        # Create masked logits using multiplication and addition
        mask_float = last_step_mask.float()
        masked_logits = logits * mask_float + (1 - mask_float) * self.neg_value
        self._debug_tensor("Masked logits", masked_logits)
        
        return masked_logits

def load_lm_model(checkpoint_path: str, device: torch.device) -> LMModel:
    """Load LM model with architecture matching the checkpoint."""
    
    pattern_provider = DelayedPatternProvider(
        n_q=4,
        delays=[0, 1, 2, 3],
        flatten_first=0,
        empty_initial=0
    )
    
    condition_provider = MinimalConditioningProvider(device)
    fuser = MinimalConditionFuser()

    # Create transformer with matching architecture
    transformer_kwargs = {
        'd_model': 1024,
        'num_heads': 16,
        'num_layers': 24,
        'dim_feedforward': 4096,
        'dropout': 0.0,
        'activation': 'gelu',
        'norm': 'layer_norm',
        'norm_first': True,
        'memory_efficient': True,
        'causal': True,
        'layer_scale': None,
        'bias_attn': False,
        'bias_proj': False,
        'bias_ff': False,
    }
    
    lm_model = LMModel(
        pattern_provider=pattern_provider,
        condition_provider=condition_provider,
        fuser=fuser,
        n_q=4,
        card=2048,
        dim=transformer_kwargs['d_model'],
        num_heads=transformer_kwargs['num_heads'],
        num_layers=transformer_kwargs['num_layers'],
        hidden_scale=4,
        norm=transformer_kwargs['norm'],
        norm_first=transformer_kwargs['norm_first'],
        emb_lr=None,
        bias_proj=transformer_kwargs['bias_proj'],
        bias_ff=transformer_kwargs['bias_ff'],
        bias_attn=transformer_kwargs['bias_attn'],
        weight_init='gaussian',
        depthwise_init='current',
        zero_bias_init=True,
        attribute_dropout={},
        two_step_cfg=False
    )
    
    # Load state dict
    state = torch.load(checkpoint_path, map_location=device)
    if "best_state" in state:
        state = state["best_state"]
    
    # Filter state dict to match model architecture
    model_state = lm_model.state_dict()
    filtered_state = {}
    
    for k, v in state.items():
        # Skip cross attention components
        if 'cross_attention' in k or 'norm_cross' in k:
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
    
    # Load with strict=True
    lm_model.load_state_dict(filtered_state, strict=True)
    lm_model.eval()
    
    return lm_model

def export_lm_model(checkpoint_path: str, output_path: str, device: torch.device = torch.device("cpu")):
    """Export LM model to ONNX format."""
    
    # Load and wrap model
    lm_model = load_lm_model(checkpoint_path, device)
    wrapper = LMInferenceWrapper(lm_model)
    wrapper.eval()
    
    # Enable debug prints for the test run
    wrapper.debug_prints = True
    
    # Create dummy input with different sequence lengths to ensure dynamic tracing
    B = 1
    K = lm_model.num_codebooks
    S = 10  # Example sequence length
    dummy_sequence = torch.zeros(B, K, S, dtype=torch.long, device=device)
    
    print("\nRunning test forward pass...")
    with torch.no_grad():
        wrapper(dummy_sequence)
    
    print("\nExporting to ONNX...")
    wrapper.debug_prints = False  # Disable debug prints for actual export
    
    # Export configuration
    input_names = ["sequence"]
    output_names = ["next_token_logits"]
    dynamic_axes = {
        "sequence": {
            0: "batch",           # Batch size
            2: "sequence"         # Sequence length
        },
        "next_token_logits": {
            0: "batch",           # Batch size
            1: "codebooks",       # Number of codebooks
            2: "vocab_size"       # Vocabulary size
        },
        # Add dynamic axes for intermediate tensors
        "attention_mask": {
            0: "batch",
            1: "sequence"
        },
        "hidden_states": {
            0: "batch",
            1: "sequence",
            2: "hidden"
        }
    }
    
    # Export to ONNX
    torch.onnx.export(
        wrapper,
        dummy_sequence,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )

if __name__ == "__main__":
    checkpoint_path = "./state_dict.bin"
    output_path = "lm_inference.onnx"
    export_lm_model(checkpoint_path, output_path)