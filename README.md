# audiocraft-onnx

Me trying (and currently failing) to convert MusicGen weights to ONNX. This has all been done with the help of robot friends, and I'm definitely in over my head.

## What's this all about?

We're trying to convert Facebook's MusicGen model (specifically using weights from [thepatch/vanya_ai_dnb_0.1](https://huggingface.co/thepatch/vanya_ai_dnb_0.1), which is a fine-tune of facebook/musicgen-small) to ONNX format so we can run it in electron, or in phones locally. Transformers.js does have a musicgen version, but there's no 'generate_continuation' function. 

this is the function we be talkin about inside the /audiocraft/models/genmodel.py:

```
def generate_continuation(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False, return_tokens: bool = False) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on audio prompts and an optional text description.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (list of str, optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)
      
```

our backend at https://github.com/betweentwomidnights/gary-backend-combined uses this function alot. It takes an input audio prompt. If we do figure this out, our applications won't need a backend in the clouds anymore.

The good news: encoding and decoding works.

The bad news: The language model part is a hellscape.

## Current Status

### What's Working
- The compression models (encode/decode) are successfully exported and working
- We can encode and decode audio with the ONNX compression models
- The LM model exports to ONNX without crashing, which was hard as f to do.

### What's Not Working
- The LM model inference is producing garbled insanity in actual use
- We think this is because of the auto-regressive loop and how ONNX graphs work
- The streaming transformer's stateful nature doesn't play nice with ONNX's static graph requirements

## The Technical Details

### Initial LM Test Output
When we run our LM test, here's what we see:

```
Running test forward pass...
Sequence mask:
Shape: torch.Size([1, 4, 10])
Dtype: torch.bool
Values: 0.850 (mean)

Sequence codes:
Shape: torch.Size([1, 4, 14])
Dtype: torch.int64
Range: [0, 2048]
Mean: 585.143
Std: 933.565

Initial logits:
Shape: torch.Size([1, 4, 14, 2048])
Dtype: torch.float32
Range: [-11.006, 16.152]
Mean: -1.047
Std: 2.214

Reshaped logits:
Shape: torch.Size([1, 4, 10, 2048])
Dtype: torch.float32
Range: [-11.006, 15.101]
Mean: -1.044
Std: 2.178

Final logits:
Shape: torch.Size([1, 4, 2048])
Dtype: torch.float32
Range: [-9.624, 5.144]
Mean: -1.047
Std: 1.938

Last step mask:
Shape: torch.Size([1, 4, 2048])
Dtype: torch.bool
Values: 1.000 (mean)

Masked logits:
Shape: torch.Size([1, 4, 2048])
Dtype: torch.float32
Range: [-9.624, 5.144]
Mean: -1.047
Std: 1.938
```

### Compression Model Testing
When we test the compression models, they work pretty well:

```
Prompt duration: 1 sec
 - Codes shape: (1, 4, 50)
 - Decoded audio shape: (1, 1, 32000)
----------------------------------------
Prompt duration: 5 sec
 - Codes shape: (1, 4, 250)
 - Decoded audio shape: (1, 1, 160000)
----------------------------------------
Prompt duration: 15 sec
 - Codes shape: (1, 4, 750)
 - Decoded audio shape: (1, 1, 480000)
```

### LM Model Comparison Results

When we compare PyTorch vs ONNX implementations using check_logits.py:

```
PyTorch logits shape: (1, 4, 2048)
ONNX logits shape:    (1, 4, 2048)
Max abs difference between PyTorch and ONNX last-step logits: 0.000238
Mean abs difference: 0.000030

Logits distribution for codebook 0:
PyTorch percentiles:
  0th: -8.964
  25th: -2.550
  50th: -1.236
  75th: 0.191
  100th: 3.940
ONNX percentiles:
  0th: -8.964
  25th: -2.550
  50th: -1.236
  75th: 0.191
  100th: 3.940

Last timestep analysis:
Valid positions in last step: 1
PyTorch top-5 tokens:
Token 176, logit=3.940
Token 1003, logit=3.903
Token 1036, logit=3.889
Token 461, logit=3.817
Token 1021, logit=3.676

ONNX top-5 tokens:
Token 176, logit=3.940
Token 1003, logit=3.903
Token 1036, logit=3.889
Token 461, logit=3.817
Token 1021, logit=3.676
```

that looks dope.

But when we look at step-by-step comparisons during inference:

```
Analyzing step 0...
  Sequence shapes match: True
  Valid positions match: True
  Special tokens match: True
  Logits shape match: True
  Max logit difference: 10.808321
  Mean logit difference: 1.419495
  PyTorch has NaN: False
  ONNX has NaN: False
  Top-k token overlap: 0/5

Analyzing step 1...
  Sequence shapes match: True
  Valid positions match: True
  Special tokens match: True
  Logits shape match: True
  Max logit difference: 0.695178
  Mean logit difference: 0.500033
  PyTorch has NaN: False
  ONNX has NaN: False
  Top-k token overlap: 2/5

Analyzing step 2...
  Sequence shapes match: True
  Valid positions match: True
  Special tokens match: True
  Logits shape match: True
  Max logit difference: 0.391212
  Mean logit difference: 0.227879
  PyTorch has NaN: False
  ONNX has NaN: False
  Top-k token overlap: 1/5

Analyzing step 3...
  Sequence shapes match: True
  Valid positions match: True
  Special tokens match: True
  Logits shape match: True
  Max logit difference: 8.896946
  Mean logit difference: 0.539138
  PyTorch has NaN: False
  ONNX has NaN: False
  Top-k token overlap: 1/5

Analyzing step 4...
  Sequence shapes match: True
  Valid positions match: True
  Special tokens match: True
  Logits shape match: True
  Max logit difference: 7.545876
  Mean logit difference: 1.265550
  PyTorch has NaN: False
  ONNX has NaN: False
  Top-k token overlap: 0/5
```

### Layer Analysis
When we analyze the model layers, we see some interesting patterns in the transformations:

```
Layer 0:
  Shape: torch.Size([1, 14, 1024])
  Mean: 0.440
  Std:  0.357
  Min:  -1.418
  Max:  4.523

[... middle layers omitted for brevity, but show gradual changes ...]

Layer 23:
  Shape: torch.Size([1, 14, 1024])
  Mean: 0.581
  Std:  0.742
  Min:  -21.428
  Max:  13.157
```

### Tracer Warnings

During the ONNX export, we get a bunch of tracer warnings. The main issues seem to be:
- Converting tensors to Python values during tracing
- Issues with dynamic axes for attention masks and hidden states
- Numpy conversions that might cause trace issues
- Problems with tensor-to-boolean conversions in various assertion checks

these warnings might be okay...we see alot of them with the encodec export, too and that works out...

note: a robot told me to export just the decoder part of encodec to its own onnx model. I'm not sure how necessary that was but I see transformers.js does similar things. 

I literally just renamed compression_encode_decode.onnx to compression_encode.onnx. Like I said, the encodec model is doin just fine. 

You'll end up with 3 onnx models:

`compression_encode.onnx`
`compression_decode.onnx`
`lm_inference.onnx`

## The Scripts

- `onnx_test.py` - Exports the encodec model (saved as compression_encode.onnx)
- `onnx_decode_test.py` - Exports just the decoder part
- `onnx_lm_test_2.py` - Attempts to export the lm model
- `check_logits.py` - Compares PyTorch vs ONNX logits
- `onnx_python_debug.py` - creates the debug_output.json and compares the pytorch/onnx model inference
- `run_onnx_encodec.py` - Tests the ONNX compression
- `analyze_checkpoint.py` - Gets details about state_dict.bin

## Why This Is Hard

The main issue is that MusicGen's language model is autoregressive - it uses its own outputs as inputs for the next step. ONNX wants a static graph, but we need something stateful. We're trying to handle the autoregressive loop at inference time instead of exporting it, but... it's not going great.

The tracings show that a lot of the model's internal logic (like assertions and tensor manipulations) isn't playing nice with ONNX's tracing system. While we can get the basic forward pass to work (as shown by the initial logit comparisons), the model starts to diverge significantly when we try to use it autoregressively.

## Help Wanted!

If you know how to handle autoregressive models in ONNX properly, please help! I'm definitely learning as I go here.