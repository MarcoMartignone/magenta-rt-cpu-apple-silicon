"""
Magenta RT Wrapper for Apple Silicon

Works around the TF 2.20 + sentencepiece mutex crash by:
1. Running sentencepiece encoding in a subprocess
2. Loading RVQ codebooks directly without importing musiccoca module

This enables full Magenta RT inference on Apple Silicon M1/M2/M3/M4.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import subprocess
import pickle
import sys
import warnings
import time
from typing import Optional, Tuple
import numpy as np


def encode_text_subprocess(text: str, vocab_path: str) -> list[int]:
    """Run sentencepiece encoding in a subprocess to avoid mutex crash."""
    # Escape quotes in text for safety
    safe_text = text.lower().replace('"', '\\"').replace("'", "\\'")
    code = f'''
import sentencepiece as spm
import pickle
import sys

vocab = spm.SentencePieceProcessor("{vocab_path}")
labels = vocab.EncodeAsIds("{safe_text}")
sys.stdout.buffer.write(pickle.dumps(labels))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f'Sentencepiece subprocess failed: {result.stderr.decode()}')
    return pickle.loads(result.stdout)


# Ordering for RVQ codebook variables (matches reference implementation)
MUSICCOCA_RVQ_VAR_ORDER = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3]


class MagentaRTAppleSilicon:
    """
    Wrapper for Magenta RT that works on Apple Silicon M1/M2/M3/M4.

    Works around the TensorFlow 2.20 mutex crash by:
    - Isolating sentencepiece operations in subprocesses
    - Loading RVQ codebooks directly without importing musiccoca module
    """

    def __init__(self, tag: str = 'large'):
        import tensorflow as tf
        import jax

        # Delayed imports to avoid loading problematic modules
        from magenta_rt import asset, spectrostream, utils
        from magenta_rt.depthformer import model

        self.tag = tag
        self._tf = tf
        self._asset = asset
        self._utils = utils

        # Configuration constants (from system.py MagentaRTConfiguration)
        self._style_embedding_dim = 768
        self._style_rvq_codebook_size = 1024
        self._encoder_style_rvq_depth = 6
        self._vocab_style_offset = 17410
        self._vocab_mask_token = 1
        self._codec_rvq_codebook_size = 1024
        self._vocab_codec_offset = 2
        self._context_tokens_shape = (250, 16)
        self._encoder_codec_rvq_depth = 4
        self._decoder_codec_rvq_depth = 16
        self._chunk_length_frames = 50
        self._crossfade_length_frames = 1
        self._frame_length_samples = 1920  # 40ms at 48kHz

        # Paths
        self._vocab_path = str(asset.fetch('vocabularies/musiccoca_mv212f_vocab.model'))
        self._encoder_path = str(asset.fetch('savedmodels/musiccoca_mv212f_cpu_novocab', is_dir=True))
        self._rvq_codebooks_path = str(asset.fetch('savedmodels/musiccoca_mv212_quant', is_dir=True))

        # Load MusicCoCa encoder (TF SavedModel - this works fine)
        print('Loading MusicCoCa encoder...', flush=True)
        with tf.device('/cpu:0'):
            self._encoder = tf.saved_model.load(self._encoder_path)

        # Load RVQ codebooks directly (bypassing musiccoca module)
        print('Loading RVQ codebooks...', flush=True)
        self._rvq_codebooks = self._load_rvq_codebooks()

        # Load SpectroStream codec
        print('Loading SpectroStream codec...', flush=True)
        self._codec = spectrostream.SpectroStreamJAX(lazy=False)

        # Load LLM checkpoint
        print('Loading LLM checkpoint (this may take a minute)...', flush=True)
        checkpoint_path = "checkpoints/llm_large_x3047_c1860k.tar" if tag == 'large' else "checkpoints/llm_base_x4286_c1860k.tar"
        checkpoint_dir = asset.fetch(checkpoint_path, is_dir=True, extract_archive=True)

        # batch_size=2 is required for classifier-free guidance (CFG)
        self._batch_size = 2

        self._task_feature_lengths, self._partitioner, self._interactive_model = (
            model.load_pretrained_model(
                checkpoint_dir=checkpoint_dir,
                size=tag,
                batch_size=self._batch_size,
                num_partitions=1,
                model_parallel_submesh=None,
            )
        )

        # Get infer function
        self._guidance_weight = 5.0
        self._temperature = 1.1
        self._topk = 40

        self._infer_fn = model.get_infer_fn(
            interactive_model=self._interactive_model,
            partitioner=self._partitioner,
            batch_size=self._batch_size,
            task_feature_lengths=self._task_feature_lengths,
            default_guidance_weight=self._guidance_weight,
            default_temperature=self._temperature,
            default_topk=self._topk,
        )

        print('✅ MagentaRT loaded successfully on Apple Silicon!', flush=True)

    def _load_rvq_codebooks(self) -> np.ndarray:
        """Load RVQ codebooks directly without importing musiccoca."""
        var_path = f'{self._rvq_codebooks_path}/variables/variables'
        result = np.zeros((12, 1024, 768), dtype=np.float32)
        for k, v_name in enumerate(MUSICCOCA_RVQ_VAR_ORDER):
            var = self._tf.train.load_variable(
                var_path, f'variables/{v_name}/.ATTRIBUTES/VARIABLE_VALUE'
            )
            result[k] = var.T
        return result

    @property
    def sample_rate(self) -> int:
        return self._codec.sample_rate

    @property
    def chunk_length(self) -> float:
        return 2.0  # seconds

    @property
    def num_channels(self) -> int:
        return 2

    @property
    def chunk_length_samples(self) -> int:
        return self._chunk_length_frames * self._frame_length_samples

    @property
    def crossfade_length_samples(self) -> int:
        return self._crossfade_length_frames * self._frame_length_samples

    def embed_style(self, text: str) -> np.ndarray:
        """Embed a text style prompt into the MusicCoCa embedding space."""
        # Run sentencepiece in subprocess to avoid mutex crash
        labels = encode_text_subprocess(text, self._vocab_path)

        # Prepare inputs
        max_length = 128
        labels = labels[:max_length - 1]
        num_tokens = len(labels)
        labels = [1] + labels + [0] * (max_length - 1 - num_tokens)

        inputs_0 = self._tf.constant([labels], dtype=self._tf.int32)
        inputs_1 = self._tf.constant(
            [[1.0] * (num_tokens + 1) + [0.0] * (max_length - 1 - num_tokens)],
            dtype=self._tf.float32
        )

        # Call encoder
        result = self._encoder.signatures['embed_text'](inputs_0=inputs_0, inputs_0_1=inputs_1)
        return result['contrastive_txt_embed_l2_normalized'].numpy()[0].astype(np.float32)

    def tokenize_style(self, embedding: np.ndarray) -> np.ndarray:
        """Tokenize a style embedding using RVQ quantization."""
        if embedding.shape != (self._style_embedding_dim,):
            raise ValueError(f"Invalid style shape: {embedding.shape}, expected ({self._style_embedding_dim},)")

        # RVQ quantization (same as utils.rvq_quantization)
        tokens, _ = self._utils.rvq_quantization(
            embedding.reshape(1, -1),
            self._rvq_codebooks
        )
        return tokens[0]  # (12,) int32

    def init_state(self) -> dict:
        """Initialize generation state."""
        return {
            'chunk_index': 0,
            'context_tokens': np.full(self._context_tokens_shape, -1, dtype=np.int32),
            'crossfade_samples': np.zeros((self.crossfade_length_samples, self.num_channels), dtype=np.float32),
        }

    def generate_chunk(
        self,
        state: Optional[dict] = None,
        style: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate a chunk of audio.

        Args:
            state: Generation state from init_state() or previous generate_chunk()
            style: Style embedding from embed_style(), shape (768,)
            seed: Random seed for generation
            **kwargs: Additional params like temperature, topk, guidance_weight

        Returns:
            Tuple of (audio samples, updated state)
        """
        import jax

        if state is None:
            state = self.init_state()
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Prepare codec tokens for LLM
        codec_tokens_lm = np.where(
            state['context_tokens'] >= 0,
            self._utils.rvq_to_llm(
                np.maximum(state['context_tokens'], 0),
                self._codec_rvq_codebook_size,
                self._vocab_codec_offset,
            ),
            np.full_like(state['context_tokens'], self._vocab_mask_token),
        )

        # Prepare style tokens for LLM
        if style is None:
            style_tokens_lm = np.full(
                (self._encoder_style_rvq_depth,),
                self._vocab_mask_token,
                dtype=np.int32,
            )
        else:
            if style.shape != (self._style_embedding_dim,):
                raise ValueError(f"Invalid style shape: {style.shape}")
            style_tokens = self.tokenize_style(style)
            style_tokens = style_tokens[:self._encoder_style_rvq_depth]  # Take first 6
            style_tokens_lm = self._utils.rvq_to_llm(
                style_tokens,
                self._style_rvq_codebook_size,
                self._vocab_style_offset,
            )

        # Prepare encoder input
        encoder_inputs_pos = np.concatenate([
            codec_tokens_lm[:, :self._encoder_codec_rvq_depth].reshape(-1),
            style_tokens_lm,
        ], axis=0)

        encoder_inputs_neg = encoder_inputs_pos.copy()
        encoder_inputs_neg[-self._encoder_style_rvq_depth:] = self._vocab_mask_token
        encoder_inputs = np.stack([encoder_inputs_pos, encoder_inputs_neg], axis=0)

        # Generate tokens via LLM
        max_decode_frames = kwargs.get('max_decode_frames', self._chunk_length_frames)

        generated_tokens, _ = self._infer_fn(
            {
                'encoder_input_tokens': encoder_inputs,
                'decoder_input_tokens': np.zeros(
                    (self._batch_size, self._chunk_length_frames * self._decoder_codec_rvq_depth),
                    dtype=np.int32,
                ),
            },
            {
                'max_decode_steps': np.array(
                    max_decode_frames * self._decoder_codec_rvq_depth,
                    dtype=np.int32,
                ),
                'guidance_weight': kwargs.get('guidance_weight', self._guidance_weight),
                'temperature': kwargs.get('temperature', self._temperature),
                'topk': kwargs.get('topk', self._topk),
            },
            jax.random.PRNGKey(seed + state['chunk_index']),
        )

        # Process generated tokens
        generated_tokens = np.array(generated_tokens)
        generated_tokens = generated_tokens[:1]  # First batch element
        generated_tokens = generated_tokens.reshape(
            self._chunk_length_frames, self._decoder_codec_rvq_depth
        )
        generated_tokens = generated_tokens[:max_decode_frames]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            generated_rvq_tokens = self._utils.llm_to_rvq(
                generated_tokens,
                self._codec_rvq_codebook_size,
                self._vocab_codec_offset,
                safe=False,
            )

        # Decode via SpectroStream with crossfade
        xfade_frames = state['context_tokens'][-self._crossfade_length_frames:]
        if state['chunk_index'] == 0:
            xfade_frames = np.zeros_like(xfade_frames)
        xfade_frames = np.maximum(xfade_frames, 0)

        xfade_tokens = np.concatenate([xfade_frames, generated_rvq_tokens], axis=0)
        chunk_with_xfade = self._codec.decode(xfade_tokens)

        # Get audio samples
        chunk_samples = chunk_with_xfade.samples

        # Perform crossfade
        xfade_samples = chunk_samples[-self.crossfade_length_samples:]
        xfade_ramp = self._make_crossfade_ramp(self.crossfade_length_samples)[:, np.newaxis]
        chunk = chunk_samples[:-self.crossfade_length_samples].copy()

        # Fade in current chunk
        chunk[:self.crossfade_length_samples] *= xfade_ramp
        # Fade out last chunk
        chunk[:self.crossfade_length_samples] += (
            state['crossfade_samples'] * np.flip(xfade_ramp, axis=0)
        )

        # Update state
        new_state = {
            'chunk_index': state['chunk_index'] + 1,
            'context_tokens': self._update_context_tokens(
                state['context_tokens'], generated_rvq_tokens
            ),
            'crossfade_samples': xfade_samples,
        }

        return chunk, new_state

    def _make_crossfade_ramp(self, length: int) -> np.ndarray:
        """Create equal-power crossfade ramp."""
        t = np.linspace(0, np.pi / 2, length)
        return np.sin(t).astype(np.float32)

    def _update_context_tokens(
        self, context: np.ndarray, new_tokens: np.ndarray
    ) -> np.ndarray:
        """Update context tokens with new generated tokens."""
        # Shift context and append new tokens
        new_context = np.full_like(context, -1)
        # Keep last (250 - len(new_tokens)) frames and add new ones
        shift = min(len(new_tokens), context.shape[0])
        if shift < context.shape[0]:
            new_context[:-shift] = context[shift:]
        new_context[-shift:, :new_tokens.shape[1]] = new_tokens[-shift:]
        return new_context


def test():
    """Test the wrapper."""
    print("=" * 60)
    print("Testing Magenta RT on Apple Silicon")
    print("=" * 60)

    # Test 1: Text embedding via subprocess
    print("\n[Test 1] Text Embedding (subprocess isolation)")
    from magenta_rt import asset
    vocab_path = str(asset.fetch('vocabularies/musiccoca_mv212f_vocab.model'))
    labels = encode_text_subprocess("upbeat electronic dance music", vocab_path)
    print(f"  ✅ Encoded: {labels[:10]}...")

    # Test 2: Full system initialization and generation
    print("\n[Test 2] Full System Test")
    rt = MagentaRTAppleSilicon(tag='large')

    print(f"\n  Sample rate: {rt.sample_rate}Hz")
    print(f"  Chunk length: {rt.chunk_length}s")
    print(f"  Channels: {rt.num_channels}")

    # Test style embedding
    print("\n[Test 3] Style Embedding")
    style = rt.embed_style("ambient electronic music with soft pads")
    print(f"  ✅ Style embedding shape: {style.shape}")
    print(f"  ✅ Style embedding sample: {style[:5]}...")

    # Test generation
    print("\n[Test 4] Audio Generation (CPU - may be slow)")
    state = rt.init_state()
    start_time = time.time()
    audio_chunk, state = rt.generate_chunk(state=state, style=style, seed=42)
    gen_time = time.time() - start_time

    print(f"  ✅ Generated audio shape: {audio_chunk.shape}")
    print(f"  ✅ Audio duration: {len(audio_chunk) / rt.sample_rate:.2f}s")
    print(f"  ✅ Generation time: {gen_time:.2f}s")
    print(f"  ✅ Real-time factor: {gen_time / rt.chunk_length:.2f}x")

    # Save audio sample
    import scipy.io.wavfile as wav
    output_path = os.path.join(os.path.dirname(__file__), 'test_output.wav')
    wav.write(output_path, rt.sample_rate, audio_chunk)
    print(f"\n  ✅ Saved audio to: {output_path}")

    print("\n" + "=" * 60)
    print("All tests passed! Magenta RT works on Apple Silicon!")
    print("=" * 60)


if __name__ == '__main__':
    test()
    os._exit(0)  # Clean exit to avoid cleanup crash
