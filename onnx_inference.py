import torch
import numpy as np
import onnxruntime as ort

# Assume onnx_single_step is available from your previous module.
from onnx_single_step import onnx_single_step

def full_streaming_generation(
    prompt_tokens_np: np.ndarray,
    lm_session: ort.InferenceSession,
    pattern_provider,
    max_gen_len: int,
    window_size: int = 10,
    temperature: float = 1.0,
    top_k: int = 250,
    special_token: int = 2047,
) -> torch.Tensor:
    """
    Fully reproduces the MusicGen LMModel.generate logic using the pattern sequence.
    
    Args:
        prompt_tokens_np: numpy array of shape [B, K, T_prompt] (encoded prompt).
        lm_session: ONNX session for the LM model.
        pattern_provider: An instance of DelayedPatternProvider.
        max_gen_len: Total generation length (number of tokens per codebook).
        window_size: Fixed window size to feed into the ONNX LM model (should match export).
        temperature: Sampling temperature.
        top_k: Top-k parameter.
        special_token: The special token id (usually equal to LMModel.card).
        
    Returns:
        A torch.Tensor of shape [B, K, T_generated] containing the full generated token sequence.
    """
    device = torch.device("cpu")  # or your desired device
    B, K, T_prompt = prompt_tokens_np.shape
    unknown_token = -1

    # Create a full token array (gen_codes) of shape [B, K, max_gen_len] filled with unknown_token.
    gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
    # Fill in the prompt tokens.
    prompt_tokens = torch.from_numpy(prompt_tokens_np).to(device)
    gen_codes[..., :T_prompt] = prompt_tokens

    # Build the pattern sequence using the pattern provider.
    pattern = pattern_provider.get_pattern(max_gen_len)
    gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, special_token)
    # gen_sequence: [B, K, S] ; mask: [K, S]
    S = gen_sequence.shape[-1]

    # Determine the starting step in the pattern based on the prompt length.
    start_offset_sequence = pattern.get_first_step_with_timesteps(T_prompt)
    if start_offset_sequence is None:
        start_offset_sequence = 0
    print(f"Starting pattern sequence generation at step {start_offset_sequence} out of {S} total steps.")

    prev_offset = 0
    # Iterate over each pattern step from the starting point to the end of the pattern.
    for offset in range(start_offset_sequence, S):
        # Extract the current sub-sequence from the pattern.
        curr_sequence = gen_sequence[..., prev_offset:offset]  # shape: [B, K, current_length]
        # Replace any unknown tokens (-1) with the special token.
        curr_sequence = torch.where(curr_sequence == unknown_token,
                                    torch.tensor(special_token, dtype=curr_sequence.dtype, device=device),
                                    curr_sequence)
        # Prepare a fixed window of length 'window_size' to feed into the ONNX model.
        current_length = curr_sequence.shape[-1]
        if current_length < window_size:
            pad_len = window_size - current_length
            pad = torch.full((B, K, pad_len), special_token, dtype=curr_sequence.dtype, device=device)
            curr_sequence_window = torch.cat([pad, curr_sequence], dim=-1)
        else:
            curr_sequence_window = curr_sequence[..., -window_size:]
        
        # Convert the current window to NumPy.
        curr_sequence_np = curr_sequence_window.cpu().numpy()
        # Call our helper to sample one new token per codebook.
        new_tokens_np, _ = onnx_single_step(
            lm_session,
            curr_sequence_np,
            step=offset,
            special_token_id=special_token,
            temperature=temperature,
            top_k=top_k,
            debug=False
        )
        new_tokens = torch.from_numpy(new_tokens_np).to(device)
        # Use the mask from the pattern to determine valid positions.
        valid_mask = mask[:, offset:offset+1].unsqueeze(0).expand(B, -1, -1)  # shape: [B, K, 1]
        # For positions not valid, force the token to be the special token.
        new_tokens = torch.where(valid_mask, new_tokens, torch.tensor(special_token, dtype=torch.long, device=device))
        # Update the gen_sequence at this offset only where the token is still unknown.
        current_val = gen_sequence[..., offset:offset+1]
        updated_val = torch.where(current_val == unknown_token, new_tokens, current_val)
        gen_sequence[..., offset:offset+1] = updated_val
        prev_offset = offset
        if offset % 10 == 0 or offset == S - 1:
            print(f"Pattern step {offset}/{S}: updated pattern sequence.")

    # Once complete, revert the pattern sequence back to the original token ordering.
    out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)
    return out_codes

# Example usage within an inference pipeline:
if __name__ == "__main__":
    import soundfile as sf
    import torchaudio
    import numpy as np
    from audiocraft.modules.codebooks_patterns import DelayedPatternProvider

    # Load ONNX sessions for LM and compression models.
    lm_session = ort.InferenceSession("lm_inference.onnx")
    encode_session = ort.InferenceSession("compression_encode.onnx")
    decode_session = ort.InferenceSession("compression_decode.onnx")

    # Audio loading and preprocessing.
    def load_audio_mono_32k(path: str):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        if sr != 32000:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, 32000)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()
            sr = 32000
        return audio.reshape(1, 1, -1), sr

    prompt_path = "prompt.wav"
    audio_np, sr = load_audio_mono_32k(prompt_path)
    # Crop or pad the prompt to 5 seconds.
    desired_len = 5 * sr
    if audio_np.shape[-1] < desired_len:
        pad = np.zeros((1, 1, desired_len - audio_np.shape[-1]), dtype=audio_np.dtype)
        prompt_audio = np.concatenate([audio_np, pad], axis=-1)
    else:
        prompt_audio = audio_np[..., :desired_len]
    
    # Encode the prompt using compression_encode.onnx.
    def encode_audio(audio_np, encode_session):
        out = encode_session.run(None, {"audio": audio_np})
        return out[0]
    prompt_tokens_np = encode_audio(prompt_audio, encode_session)  # shape: [B, K, T_prompt]
    print("Encoded prompt tokens shape:", prompt_tokens_np.shape)

    # Create a DelayedPatternProvider (using same parameters as in MusicGen).
    pattern_provider = DelayedPatternProvider(n_q=4, delays=[0, 1, 2, 3], flatten_first=0, empty_initial=0)
    max_gen_len = 500  # Total tokens per codebook.

    # Run the full streaming generation.
    generated_tokens = full_streaming_generation(
        prompt_tokens_np,
        lm_session,
        pattern_provider,
        max_gen_len,
        window_size=10,
        temperature=1.0,
        top_k=250,
        special_token=2047
    )
    print("Generated token sequence shape:", generated_tokens.shape)

    # Decode the tokens back into audio.
    def decode_tokens(tokens_np, decode_session):
        out = decode_session.run(None, {"codes": tokens_np})
        return out[0]
    generated_tokens_np = generated_tokens.cpu().numpy()
    generated_audio = decode_tokens(generated_tokens_np, decode_session)
    print("Decoded generated audio shape:", generated_audio.shape)

    sf.write("continuation_full_streaming.wav", generated_audio[0, 0], sr)
    print("Saved generated audio to continuation_full_streaming.wav")
