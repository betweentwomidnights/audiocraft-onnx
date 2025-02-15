import numpy as np
from typing import Tuple

def onnx_single_step(
    lm_session,
    current_sequence: np.ndarray,
    step: int,
    special_token_id: int = 2047,
    temperature: float = 1.0,
    top_k: int = 250,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs one autoregressive step using the ONNX LM model.
    
    Args:
        lm_session: The ONNX Runtime session for the LM model.
        current_sequence: np.ndarray of shape [B, K, T] with current tokens (dtype=np.int64).
        step: Current autoregressive step (int). Determines which codebooks should generate.
        special_token_id: The token ID used for codebooks that have not yet generated.
        temperature: Sampling temperature.
        top_k: Top-k parameter for sampling.
        debug: If True, prints debug information.
        
    Returns:
        next_tokens: np.ndarray of shape [B, K, 1] containing the new tokens.
        logits: np.ndarray of shape [B, K, vocab_size] from the LM model.
    """
    # Ensure input is int64
    current_sequence = current_sequence.astype(np.int64)
    
    # Run the LM ONNX model.
    # It should accept "sequence" as input and return logits for the last timestep.
    outputs = lm_session.run(None, {"sequence": current_sequence})
    logits = outputs[0]  # Expected shape: [B, K, vocab_size]
    
    if debug:
        B, K, vocab_size = logits.shape
        print(f"Step {step}: logits shape: {logits.shape}")
        print(f"Logits stats: min={logits.min():.3f}, max={logits.max():.3f}, "
              f"mean={logits.mean():.3f}, std={logits.std():.3f}")
    
    B, K, vocab_size = logits.shape
    next_tokens = np.empty((B, K, 1), dtype=np.int64)
    
    # For each batch and each codebook, sample if allowed by delayed pattern.
    for b in range(B):
        for k in range(K):
            if step >= k:
                # Copy logits for current batch and codebook.
                logits_bk = logits[b, k, :].copy()
                # Temperature scaling
                if temperature != 1.0:
                    logits_bk = logits_bk / temperature

                # Top-k filtering: set logits outside top-k to -infinity.
                if top_k > 0:
                    # Identify indices that are NOT in the top-k.
                    indices_to_remove = np.argpartition(logits_bk, -top_k)[:-top_k]
                    logits_bk[indices_to_remove] = -np.inf

                # Compute softmax over filtered logits.
                max_logit = np.max(logits_bk)
                exp_logits = np.exp(logits_bk - max_logit)
                probs = exp_logits / np.sum(exp_logits)
                # Sample a token from the probability distribution.
                sampled_token = np.random.choice(vocab_size, p=probs)
                if debug:
                    print(f"Batch {b}, Codebook {k}: sampled token {sampled_token} "
                          f"(prob: {probs[sampled_token]:.4f})")
            else:
                # For delayed codebooks, assign the special token.
                sampled_token = special_token_id
                if debug:
                    print(f"Batch {b}, Codebook {k}: delayed (step {step} < {k}), "
                          f"using special token {special_token_id}")
            next_tokens[b, k, 0] = sampled_token

    return next_tokens, logits


# Example usage inside an autoregressive loop:
if __name__ == "__main__":
    import onnxruntime as ort

    # Assume you have a working ONNX LM model session
    lm_session = ort.InferenceSession("lm_inference.onnx")
    
    # Example initial prompt tokens (shape [B, K, T_prompt])
    # For instance, B=1, K=4, T_prompt=10
    prompt_tokens = np.zeros((1, 4, 10), dtype=np.int64)
    
    # We'll build the generated sequence by starting with the prompt
    generated_sequence = prompt_tokens.copy()
    
    # Define generation parameters
    max_gen_steps = 50  # number of autoregressive steps to perform
    special_token_id = 2047  # adjust based on your model's valid token range
    temperature = 1.0
    top_k = 250

    # Autoregressive loop
    for step in range(max_gen_steps):
        new_tokens, _ = onnx_single_step(
            lm_session,
            generated_sequence,
            step=step,
            special_token_id=special_token_id,
            temperature=temperature,
            top_k=top_k,
            debug=(step in [0, max_gen_steps // 2, max_gen_steps - 1])  # debug at key steps
        )
        # Append new tokens to the sequence along the time axis.
        generated_sequence = np.concatenate([generated_sequence, new_tokens], axis=-1)
        print(f"Completed step {step + 1}/{max_gen_steps}: sequence shape {generated_sequence.shape}")

    # After the loop, generated_sequence has shape [B, K, T_prompt + max_gen_steps]
    print("Final generated sequence shape:", generated_sequence.shape)
