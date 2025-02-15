#!/usr/bin/env python3
import torch
from audiocraft.models.lm import LMModel
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from collections import defaultdict
import json

def analyze_checkpoint(checkpoint_path: str):
    """Analyze the keys and shapes in a checkpoint file."""
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location='cpu')
    if "best_state" in state:
        state = state["best_state"]
    
    # Organize keys by layer/component
    organized_keys = defaultdict(list)
    shapes = {}
    
    for key in state.keys():
        # Split the key into parts
        parts = key.split('.')
        
        # Get the main component (first part of the key)
        main_component = parts[0]
        organized_keys[main_component].append(key)
        
        # Store tensor shape
        shapes[key] = tuple(state[key].shape)
    
    # Print analysis
    print("\n=== Checkpoint Analysis ===")
    print(f"\nTotal number of keys: {len(state)}")
    
    print("\n=== Keys by Component ===")
    for component, keys in organized_keys.items():
        print(f"\n{component}: {len(keys)} keys")
        for key in sorted(keys):
            print(f"  {key}: {shapes[key]}")
    
    # Special attention to transformer layers
    if 'transformer' in organized_keys:
        print("\n=== Transformer Layer Analysis ===")
        # Get number of layers
        layer_nums = set()
        for key in organized_keys['transformer']:
            parts = key.split('.')
            if len(parts) > 2 and parts[1] == 'layers':
                try:
                    layer_nums.add(int(parts[2]))
                except ValueError:
                    continue
        
        num_layers = len(layer_nums)
        print(f"Number of transformer layers: {num_layers}")
        
        # Analyze first layer in detail
        print("\nDetailed analysis of first transformer layer:")
        first_layer_keys = [k for k in organized_keys['transformer'] 
                          if '.layers.0.' in k]
        for key in sorted(first_layer_keys):
            print(f"  {key}: {shapes[key]}")
    
    return state, organized_keys, shapes

def create_and_analyze_model():
    """Create a fresh LM model and analyze its state dict."""
    print("\n=== Creating fresh LM model for comparison ===")
    
    pattern_provider = DelayedPatternProvider(
        n_q=4,
        delays=[0, 1, 2, 3],
        flatten_first=0,
        empty_initial=0
    )
    
    # Create minimal model for comparison
    model = LMModel(
        pattern_provider=pattern_provider,
        condition_provider=None,
        fuser=None,
        n_q=4,
        card=2048,
        dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_scale=4,
        norm='layer_norm',
        norm_first=True,
        bias_proj=False,
        bias_ff=False,
        bias_attn=False
    )
    
    # Get state dict
    state = model.state_dict()
    
    # Organize keys
    organized_keys = defaultdict(list)
    shapes = {}
    
    for key in state.keys():
        parts = key.split('.')
        main_component = parts[0]
        organized_keys[main_component].append(key)
        shapes[key] = tuple(state[key].shape)
    
    print("\n=== Fresh Model State Dict Analysis ===")
    for component, keys in organized_keys.items():
        print(f"\n{component}: {len(keys)} keys")
        for key in sorted(keys):
            print(f"  {key}: {shapes[key]}")
    
    return state, organized_keys, shapes

def compare_state_dicts(checkpoint_data, model_data):
    """Compare checkpoint and fresh model state dicts."""
    checkpoint_state, checkpoint_keys, checkpoint_shapes = checkpoint_data
    model_state, model_keys, model_shapes = model_data
    
    print("\n=== State Dict Comparison ===")
    
    # Find missing and extra keys
    checkpoint_key_set = set(checkpoint_state.keys())
    model_key_set = set(model_state.keys())
    
    missing_keys = model_key_set - checkpoint_key_set
    extra_keys = checkpoint_key_set - model_key_set
    
    print("\nKeys missing from checkpoint:")
    for key in sorted(missing_keys):
        print(f"  {key}: {model_shapes[key]}")
    
    print("\nExtra keys in checkpoint:")
    for key in sorted(extra_keys):
        print(f"  {key}: {checkpoint_shapes[key]}")
    
    # Compare shapes of common keys
    print("\nShape mismatches for common keys:")
    common_keys = checkpoint_key_set & model_key_set
    for key in sorted(common_keys):
        if checkpoint_shapes[key] != model_shapes[key]:
            print(f"  {key}:")
            print(f"    Checkpoint: {checkpoint_shapes[key]}")
            print(f"    Model: {model_shapes[key]}")

if __name__ == "__main__":
    checkpoint_path = "./state_dict.bin"
    
    # Analyze checkpoint
    checkpoint_data = analyze_checkpoint(checkpoint_path)
    
    # Create and analyze fresh model
    model_data = create_and_analyze_model()
    
    # Compare them
    compare_state_dicts(checkpoint_data, model_data)