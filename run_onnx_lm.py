#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort

def test_lm_onnx_model(onnx_model_path, num_codebooks, vocab_size, test_lengths):
    """
    Tests an exported LM ONNX model by feeding random token sequences of different lengths.
    
    Args:
        onnx_model_path (str): Path to the exported ONNX file (e.g. "lm_inference.onnx").
        num_codebooks (int): K, the number of codebooks (e.g., 4 for n_q=4).
        vocab_size (int): The vocabulary size (card). E.g., 2048.
        test_lengths (list of int): Various sequence lengths (S) to test.
    """
    # Create an ONNX inference session (on CPU here).
    session = ort.InferenceSession(onnx_model_path)

    for seq_len in test_lengths:
        # Build a random sequence of shape [B=1, K=num_codebooks, S=seq_len].
        # We assume token IDs are in [0, vocab_size).
        dummy_input = np.random.randint(0, vocab_size, (1, num_codebooks, seq_len), dtype=np.int64)

        # Run inference (the wrapper has only one input named "sequence").
        outputs = session.run(None, {"sequence": dummy_input})
        next_logits = outputs[0]

        # next_logits should have shape [B=1, K=num_codebooks, card=vocab_size].
        print(f"Input sequence length: {seq_len}")
        print(f" - next_logits shape: {next_logits.shape}")
        # Check that the shape matches [1, num_codebooks, vocab_size].
        assert next_logits.shape == (1, num_codebooks, vocab_size), \
            f"Unexpected output shape: {next_logits.shape}"
        print("-" * 40)

if __name__ == "__main__":
    onnx_model_path = "lm_inference.onnx"  # Path to your exported LM model
    num_codebooks = 4   # Update to match the model's n_q
    vocab_size = 2048   # Update to match the model's card
    test_lengths = [10, 25, 50]  # Try a few sequence lengths

    test_lm_onnx_model(onnx_model_path, num_codebooks, vocab_size, test_lengths)
