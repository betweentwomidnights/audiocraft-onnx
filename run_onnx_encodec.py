#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort

def test_onnx_model(onnx_model_path, sample_rate, channels, prompt_durations):
    # Load the ONNX model session (using CPU here)
    session = ort.InferenceSession(onnx_model_path)

    for duration in prompt_durations:
        # Compute the number of time steps given the sample rate
        T = int(sample_rate * duration)
        # Create a dummy input audio signal: shape [Batch, Channels, Time]
        dummy_audio = np.random.randn(1, channels, T).astype(np.float32)
        
        # Run the model: our exported model expects an input named "audio"
        inputs = {"audio": dummy_audio}
        outputs = session.run(None, inputs)
        codes, decoded_audio = outputs

        print(f"Prompt duration: {duration} sec")
        print(f" - Codes shape: {codes.shape}")
        print(f" - Decoded audio shape: {decoded_audio.shape}")
        print("-" * 40)

if __name__ == '__main__':
    onnx_model_path = "compression_encode.onnx"  # Path to your exported ONNX model
    sample_rate = 32000  # Adjust to match the sample rate of your compression model
    channels = 1         # Typically mono audio; adjust if necessary
    prompt_durations = [1, 5, 15]  # Test with 1s, 5s, and 15s prompts

    test_onnx_model(onnx_model_path, sample_rate, channels, prompt_durations)
