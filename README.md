# SparseGPT

This repository contains code to reproduce the key results of the paper [SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot](https://arxiv.org/abs/2301.00774).

Specifically, it provides scripts and implementations to:

* Evaluate baseline and pruned models on raw-WikiText2, PTB and C4-subset. (`datautils.py`, `opt.py`, `bloom.py`) 
* Perform unstructured, n:m and sparse + quantized SparseGPT compression on OPT and BLOOM models. (`sparsegpt.py`, `opt.py`, `bloom.py`)

We note that this SparseGPT implementation is based on our open-source [GPTQ code](https://github.com/IST-DASLab/gptq). 

## Dependencies

Install the current dependencies with `pip install -r requirements.txt`. Modern
Transformers is required for Llama 3, Mistral, and Qwen2.5 support.

## Supported language models

The architecture-specific scripts support these Hugging Face model types:

| Family | Script | `config.model_type` | Example checkpoint |
| --- | --- | --- | --- |
| Llama 2 and Llama 3 | `llama.py` | `llama` | `meta-llama/Meta-Llama-3-8B` |
| Mistral | `mistral.py` | `mistral` | `mistralai/Mistral-7B-v0.1` |
| Qwen2.5 | `qwen.py` | `qwen2` (Hugging Face identifier) | `Qwen/Qwen2.5-7B` |
| GPT-2 | `gpt2.py` | `gpt2` | `openai-community/gpt2` |

This includes the requested Llama 2 7B/70B, Llama 3 8B, Mistral 7B,
Qwen2.5 7B, and GPT-2 Small checkpoints. For Qwen, the supported target is
specifically `Qwen/Qwen2.5-7B`.

The implementation prunes dense projections inside transformer blocks. Token
embeddings, normalization layers, and the language-model head remain dense.
It retains SparseGPT's one-shot, layer-by-layer algorithm without fine-tuning or
adding a different mask-selection objective.

## Usage

Use the model-specific script for each architecture:

```
# Llama 3 8B, 50% unstructured sparsity
python llama.py meta-llama/Meta-Llama-3-8B c4 --sparsity .5

# Mistral 7B, 2:4 sparsity
python mistral.py mistralai/Mistral-7B-v0.1 c4 --prunen 2 --prunem 4

# Qwen2.5 7B
python qwen.py Qwen/Qwen2.5-7B c4 --sparsity .5

# GPT-2 Small
python gpt2.py openai-community/gpt2 c4 --sparsity .5
```

Calibration defaults remain 128 samples of up to 2048 tokens, 1% Hessian dampening, and a pruning block size of 128. Use `--seqlen` to select a shorter calibration length for smoke tests. Model access approval is required for gated Meta checkpoints. Llama 2 70B also requires enough CPU memory to hold the model while one transformer block at a time is processed on the accelerator.

The older architecture-specific scripts below remain available for reproducing the original repository commands.

## How the new model drivers work

`mistral.py`, `qwen.py`, and `gpt2.py` follow the original repository's
one-file-per-architecture style. Each file defines its model paths and
true-sequential projection groups, then calls the repeated workflow in the
lightweight `llmutils.py` helper. The existing `llama.py`, `opt.py`, and
`bloom.py` files are unchanged.

Adding the new drivers did **not** change the pruning algorithm in
`sparsegpt.py`. In particular, the following SparseGPT operations are unchanged:

1. Forward hooks collect each selected layer's calibration inputs.
2. `SparseGPT.add_batch` accumulates the input-based approximate Hessian.
3. `SparseGPT.fasterprune` adds the configured diagonal dampening (1% by
   default), computes the inverse-Hessian factor using Cholesky operations, and
   processes weights in blocks of 128 columns by default.
4. The existing Hessian-aware score selects either the requested fraction of
   weights for unstructured pruning or N weights in every group of M.
5. Each removed weight's local error is propagated to the remaining weights in
   the block and then to later blocks of columns.
6. The updated block output becomes the input calibration data for the next
   transformer block. There is no training, gradient descent, or recovery
   fine-tuning.

The shared helper changes the surrounding model-specific process as follows:

- It uses `AutoConfig` and `AutoModelForCausalLM`; each model-specific file
  supplies the expected Hugging Face model type and architecture paths.
- It locates the transformer blocks, input embeddings, final normalization, and
  language-model head through architecture-specific paths.
- It captures and replays all positional and keyword arguments passed to the
  first transformer block. This accommodates rotary-position inputs, causal
  masks, and other arguments used by current Transformers versions.
- It processes one transformer block on the selected device at a time and moves
  the completed block back to CPU, preserving the original layer-wise memory
  strategy.
- It derives the calibration sequence length from the model configuration,
  capped at the paper-style default of 2048 tokens, unless `--seqlen` is given.
- It adds argument validation, complete transformer-block sparsity reporting,
  reusable tokenizer loading, and tokenizer saving alongside model weights.
- It adds discovery of Hugging Face `Conv1D` modules. This is required for
  GPT-2; `sparsegpt.py` already contained the corresponding transpose logic.
- Perplexity evaluation uses the number of predicted next-token labels
  (`sequence_length - 1`) in its normalization. This affects only reporting,
  not pruning or mask selection.

The new drivers do not include the older scripts' magnitude-pruning or
W&B logging paths. Optional weight quantization remains available through
`--wbits`, but the default `--wbits 16` performs sparsification only.

### Weights pruned by default

Only dense projection weights inside every transformer block are passed to
SparseGPT. The default selection is:

| Model family | Pruned module names in each transformer block |
| --- | --- |
| Llama 2/3 | `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` |
| Mistral | `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` |
| Qwen2.5 | `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` |
| GPT-2 | `attn.c_attn` (combined Q/K/V), `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj` |

SparseGPT changes each selected module's `weight` tensor. Biases are not
pruned. The following parameters remain dense:

- Token and positional embeddings
- Rotary-position modules
- Normalization layers
- Bias tensors
- The final language-model head
- Any parameter outside the transformer-block list

With no layer filters, `--sparsity .5` applies an approximately 50% mask to
each discovered projection. `--prunen 2 --prunem 4` instead applies exactly two
zeros per group of four along SparseGPT's input-column dimension. `--minlayer`,
`--maxlayer`, `--prune-only`, and `--invert` can restrict this default scope.

With `--true-sequential`, projections are calibrated and pruned in dependency
groups. Llama, Mistral, and Qwen2.5 use Q/K/V, attention output, MLP up/gate,
and MLP down groups. GPT-2 uses combined Q/K/V, attention output, MLP input,
and MLP output groups. Without the flag, all selected projections in a
transformer block collect statistics from the same block input pass, matching
the older scripts' default behavior.

Here are some sample commands to run baselines and sparsification on OPT models, followed by perplexity evaluations on raw-WikiText2, PTB and C4.
See also the CMD-argument documentation.

```
# Run dense baseline
python opt.py facebook/opt-125m c4

# Run magnitude baseline
python opt.py facebook/opt-125m c4 --sparsity .5 --gmp

# Prune to 50\% uniform sparsity with SparseGPT
python opt.py facebook/opt-125m c4 --sparsity .5

# Prune to full 2:4 sparsity with SparseGPT
python opt.py facebook/opt-125m c4 --prunen 2 --prunem 4

# Prune to 50\% + 4-bit with SparseGPT
python opt.py facebook/opt-125m c4 --sparsity .5 --wbits 4
```

To run on other OPT models, replace "facebook/opt-125m" by the HuggingFace name of the corresponding model.
For the 175B model, access must first be requested from Meta and the checkpoint converted to HuggingFace format, then its location can simply be passed as a name to this script.

The BLOOM script `bloom.py` has a very similar interface, however some features are currently only available for OPT, e.g.:

```
# Sparsify BLOOM-176B with SparseGPT
python bloom.py bigscience/bloom c4 --sparsity .5
```

We also provide LLaMA pruning script with the very same interface:

```
# Sparsify LLaMa with SparseGPT
python llama.py LLAMA_HF_WEIGHTS_LOCATION c4 --sparsity 0.5
```

In case one would like to save the sparsified model specify path to saved checkpoint via  `--save` flag.

One can optionally log evalution results to W&B with `--log_wandb`. 

## Demo

One can try SparseGPT via the colab demo - `demo.ipynb`. 

## Cite

If you found this work useful, please consider citing:

```
@article{frantar-sparsegpt,
  title={{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot}, 
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint arXiv:2301.00774}
}
```
