# Magenta RT on Apple Silicon

Run [Google's Magenta RT](https://github.com/magenta/magenta-realtime) music generation on Apple Silicon.

## ⚠️ CPU-Only Disclaimer

This runs on **CPU only**, not Metal GPU. Why?

- **JAX has no Metal backend**: Magenta RT uses JAX for LLM inference, and JAX only supports NVIDIA GPUs. There is no Metal/MPS support for JAX on macOS - at this point in time -
- **TensorFlow-Metal doesn't help**: While TensorFlow has a Metal plugin, LLM runs through JAX, which can't use it.

This means generation is slow (~30-60 sec per 2-second chunk). For faster generation, use an NVIDIA GPU or wait for Google to add Metal support to JAX lol 🙃

## The Problem

Magenta RT crashes on Apple Silicon due to a [protobuf symbol conflict](https://github.com/tensorflow/tensorflow/issues/98563) between TensorFlow 2.20 and sentencepiece. The error looks like:

```
libc++abi: terminating due to uncaught exception of type
std::__1::system_error: mutex lock failed: Invalid argument
```

## The Solution

This wrapper isolates sentencepiece in a subprocess and loads RVQ codebooks directly, avoiding the conflict entirely.

## Setup

```bash
# Clone this repo
git clone https://github.com/MarcoMartignone/magenta-rt-cpu-apple-silicon.git
cd magenta-rt-cpu-apple-silicon

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from magenta_rt_wrapper import MagentaRTAppleSilicon
import scipy.io.wavfile as wav

# Initialize (takes ~15-30 seconds)
rt = MagentaRTAppleSilicon(tag='large')  # or 'base' for smaller model

# Create style from text prompt
style = rt.embed_style("uk jungle break")

# Generate 2-second audio chunks
state = rt.init_state()
audio, state = rt.generate_chunk(state=state, style=style, seed=42)

# Generate more continuous audio
audio2, state = rt.generate_chunk(state=state, style=style)

# Save to file
wav.write('output.wav', rt.sample_rate, audio)
```

Run the test:

```bash
python magenta_rt_wrapper.py
```

## Performance

| Metric | Value |
|--------|-------|
| Model load | ~15-30 sec |
| First chunk (XLA compile) | ~2-5 min |
| Subsequent chunks | ~30-60 sec (CPU) |
| Chunk duration | 2 sec stereo @ 48kHz |

## Parameters

- `tag`: `'large'` (800M params) or `'base'` (smaller)
- `style`: Embedding from `embed_style(text)`
- `seed`: Random seed for reproducibility
- `temperature`: Sampling temp (default: 1.1)
- `topk`: Top-k sampling (default: 40)
- `guidance_weight`: CFG weight (default: 5.0)

## How It Works

1. **Subprocess isolation**: Sentencepiece runs in a separate process, avoiding the protobuf conflict
2. **Direct codebook loading**: RVQ codebooks are loaded without importing the problematic `musiccoca` module
3. **Classifier-free guidance**: Uses batch_size=2 for CFG (positive + negative prompts)

## Related Issues

- [tensorflow/tensorflow#98563](https://github.com/tensorflow/tensorflow/issues/98563)
- [apache/arrow#40088](https://github.com/apache/arrow/issues/40088)
- [magenta/magenta-realtime#12](https://github.com/magenta/magenta-realtime/issues/12)

## License

MIT - See the [Magenta RT repo](https://github.com/magenta/magenta-realtime) for model licensing.
