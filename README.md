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

The unified `llm.py` entry point supports these Hugging Face model types:

| Family | `config.model_type` | Example checkpoint |
| --- | --- | --- |
| Llama 2 and Llama 3 | `llama` | `meta-llama/Meta-Llama-3-8B` |
| Mistral | `mistral` | `mistralai/Mistral-7B-v0.1` |
| Qwen2.5 | `qwen2` (Hugging Face identifier) | `Qwen/Qwen2.5-7B` |
| GPT-2 | `gpt2` | `openai-community/gpt2` |

This includes the requested Llama 2 7B/70B, Llama 3 8B, Mistral 7B,
Qwen2.5 7B, and GPT-2 Small checkpoints. For Qwen, the supported target is
specifically `Qwen/Qwen2.5-7B`.

The implementation prunes dense projections inside transformer blocks. Token
embeddings, normalization layers, and the language-model head remain dense.
It retains SparseGPT's one-shot, layer-by-layer algorithm without fine-tuning or
adding a different mask-selection objective.

## Usage

Use `llm.py` for all newly supported language models:

```
# Llama 3 8B, 50% unstructured sparsity
python llm.py meta-llama/Meta-Llama-3-8B c4 --sparsity .5

# Mistral 7B, 2:4 sparsity
python llm.py mistralai/Mistral-7B-v0.1 c4 --prunen 2 --prunem 4

# Qwen2.5 7B
python llm.py Qwen/Qwen2.5-7B c4 --sparsity .5

# GPT-2 Small
python llm.py openai-community/gpt2 c4 --sparsity .5
```

Calibration defaults remain 128 samples of up to 2048 tokens, 1% Hessian dampening, and a pruning block size of 128. Use `--seqlen` to select a shorter calibration length for smoke tests. Model access approval is required for gated Meta checkpoints. Llama 2 70B also requires enough CPU memory to hold the model while one transformer block at a time is processed on the accelerator.

The older architecture-specific scripts below remain available for reproducing the original repository commands.

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
