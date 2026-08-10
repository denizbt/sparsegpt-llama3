"""SparseGPT driver for supported decoder-only language models.

The architecture adapter only describes how to traverse a Hugging Face model.
The pruning objective and update rule remain implemented by ``SparseGPT``.
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from datautils import get_loaders
from modelutils import DEV, empty_cache, find_layers
from quant import Quantizer
from sparsegpt import SparseGPT


@dataclass(frozen=True)
class Architecture:
    block_path: str
    input_paths: tuple[str, ...]
    final_norm_path: str
    head_path: str
    sequential_groups: tuple[tuple[str, ...], ...]


LLAMA_LIKE_GROUPS = (
    ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"),
    ("self_attn.o_proj",),
    ("mlp.up_proj", "mlp.gate_proj"),
    ("mlp.down_proj",),
)

ARCHITECTURES = {
    "llama": Architecture(
        "model.layers", ("model.embed_tokens", "model.rotary_emb"),
        "model.norm", "lm_head", LLAMA_LIKE_GROUPS,
    ),
    "mistral": Architecture(
        "model.layers", ("model.embed_tokens", "model.rotary_emb"),
        "model.norm", "lm_head", LLAMA_LIKE_GROUPS,
    ),
    # Transformers uses model_type="qwen2" for Qwen2.5 checkpoints.
    "qwen2": Architecture(
        "model.layers", ("model.embed_tokens", "model.rotary_emb"),
        "model.norm", "lm_head", LLAMA_LIKE_GROUPS,
    ),
    "gpt2": Architecture(
        "transformer.h", ("transformer.wte", "transformer.wpe"),
        "transformer.ln_f", "lm_head",
        (
            ("attn.c_attn",),
            ("attn.c_proj",),
            ("mlp.c_fc",),
            ("mlp.c_proj",),
        ),
    ),
}


def resolve_attr(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_attr(obj, path, value):
    parts = path.split(".")
    parent = resolve_attr(obj, ".".join(parts[:-1])) if len(parts) > 1 else obj
    setattr(parent, parts[-1], value)


def get_architecture(model_type):
    try:
        return ARCHITECTURES[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(
            f"Unsupported model type {model_type!r}; supported types: {supported}"
        ) from exc


def infer_seqlen(config, requested=None):
    if requested is not None:
        return requested
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return min(value, 2048)
    return 2048


def load_model(model_name, seqlen=None):
    config = AutoConfig.from_pretrained(model_name)
    architecture = get_architecture(config.model_type)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", low_cpu_mem_usage=True
    )
    model.seqlen = infer_seqlen(model.config, seqlen)
    return model, architecture


def move_optional(model, paths, device):
    moved = []
    for path in paths:
        try:
            module = resolve_attr(model, path)
        except AttributeError:
            continue
        if module is not None:
            set_attr(model, path, module.to(device))
            moved.append(path)
    return moved


@torch.no_grad()
def capture_inputs(model, architecture, batches, nsamples, device):
    blocks = resolve_attr(model, architecture.block_path)
    moved_paths = move_optional(model, architecture.input_paths, device)
    blocks[0] = blocks[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inputs = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {"index": 0, "args": (), "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, *args, **kwargs):
            index = cache["index"]
            if index >= nsamples:
                raise StopIteration
            inputs[index].copy_(hidden_states.squeeze(0))
            cache["index"] += 1
            cache["args"] = args
            cache["kwargs"] = kwargs
            raise StopIteration

    blocks[0] = Catcher(blocks[0])
    try:
        for batch in batches:
            if cache["index"] >= nsamples:
                break
            try:
                model(batch[0].to(device))
            except StopIteration:
                pass
    finally:
        blocks[0] = blocks[0].module

    blocks[0] = blocks[0].cpu()
    move_optional(model, moved_paths, torch.device("cpu"))
    empty_cache(device)

    if cache["index"] != nsamples:
        raise RuntimeError(f"Captured {cache['index']} samples, expected {nsamples}")
    if cache["kwargs"] is None:
        raise RuntimeError("The first transformer block did not receive any inputs")
    return inputs, cache["args"], cache["kwargs"]


def selected(name, block_index, args):
    included = args.minlayer <= block_index < args.maxlayer and args.prune_only in name
    return included != args.invert


@torch.no_grad()
def prune_model(model, architecture, dataloader, args, device=DEV):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    blocks = resolve_attr(model, architecture.block_path)
    inputs, block_args, block_kwargs = capture_inputs(
        model, architecture, dataloader, args.nsamples, device
    )
    outputs = torch.zeros_like(inputs)

    try:
        for block_index in range(len(blocks)):
            print(f"Block {block_index}/{len(blocks) - 1}")
            block = blocks[block_index].to(device)
            all_layers = find_layers(block)
            if args.true_sequential:
                groups = architecture.sequential_groups
            else:
                groups = (tuple(all_layers),)

            for names in groups:
                subset = {
                    name: all_layers[name]
                    for name in names
                    if name in all_layers and selected(name, block_index, args)
                }
                if not subset:
                    continue

                pruners = {name: SparseGPT(layer) for name, layer in subset.items()}
                if args.wbits < 16:
                    for pruner in pruners.values():
                        pruner.quantizer = Quantizer()
                        pruner.quantizer.configure(
                            args.wbits, perchannel=True, sym=False, mse=False
                        )

                handles = []
                for name, layer in subset.items():
                    def add_batch(_, hook_inputs, hook_output, name=name):
                        pruners[name].add_batch(hook_inputs[0].data, hook_output.data)
                    handles.append(layer.register_forward_hook(add_batch))

                for sample in range(args.nsamples):
                    outputs[sample] = block(
                        inputs[sample].unsqueeze(0), *block_args, **block_kwargs
                    )[0]
                for handle in handles:
                    handle.remove()

                for name, pruner in pruners.items():
                    print(f"Pruning block {block_index}: {name}")
                    pruner.fasterprune(
                        args.sparsity, prunen=args.prunen, prunem=args.prunem,
                        percdamp=args.percdamp, blocksize=args.blocksize,
                    )
                    pruner.free()

            for sample in range(args.nsamples):
                outputs[sample] = block(
                    inputs[sample].unsqueeze(0), *block_args, **block_kwargs
                )[0]
            blocks[block_index] = block.cpu()
            del block
            empty_cache(device)
            inputs, outputs = outputs, inputs
    finally:
        model.config.use_cache = use_cache


@torch.no_grad()
def evaluate(model, architecture, test_encoding, device=DEV):
    input_ids = test_encoding.input_ids
    nsamples = input_ids.numel() // model.seqlen
    if nsamples == 0:
        raise ValueError("Evaluation data is shorter than one model sequence")

    batches = [
        (input_ids[:, i * model.seqlen:(i + 1) * model.seqlen], None)
        for i in range(nsamples)
    ]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    blocks = resolve_attr(model, architecture.block_path)
    inputs, block_args, block_kwargs = capture_inputs(
        model, architecture, batches, nsamples, device
    )
    outputs = torch.zeros_like(inputs)

    final_norm = None
    head = None
    try:
        for index in range(len(blocks)):
            block = blocks[index].to(device)
            for sample in range(nsamples):
                outputs[sample] = block(
                    inputs[sample].unsqueeze(0), *block_args, **block_kwargs
                )[0]
            blocks[index] = block.cpu()
            del block
            empty_cache(device)
            inputs, outputs = outputs, inputs

        final_norm = resolve_attr(model, architecture.final_norm_path).to(device)
        head = resolve_attr(model, architecture.head_path).to(device)
        labels = input_ids.to(device)
        nlls = []
        loss_fn = nn.CrossEntropyLoss()
        for index in range(nsamples):
            logits = head(final_norm(inputs[index].unsqueeze(0)))
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[
                :, index * model.seqlen:(index + 1) * model.seqlen
            ][:, 1:]
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
            )
            nlls.append(loss.float() * (model.seqlen - 1))
        perplexity = torch.exp(torch.stack(nlls).sum() / (nsamples * (model.seqlen - 1)))
        return perplexity.item()
    finally:
        if final_norm is not None:
            set_attr(model, architecture.final_norm_path, final_norm.cpu())
        if head is not None:
            set_attr(model, architecture.head_path, head.cpu())
        empty_cache(device)
        model.config.use_cache = use_cache


def report_sparsity(model, architecture):
    zero = total = 0
    for block in resolve_attr(model, architecture.block_path):
        for layer in find_layers(block).values():
            zero += torch.count_nonzero(layer.weight == 0).item()
            total += layer.weight.numel()
    ratio = zero / total if total else 0.0
    print(f"Transformer-block sparsity: {ratio:.6f} ({zero}/{total})")
    return ratio


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model name or local path")
    parser.add_argument("dataset", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--wbits", type=int, default=16)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--prune-only", default="")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--save", default="")
    parser.add_argument(
        "--eval-datasets", nargs="*", default=["wikitext2", "ptb", "c4"],
        choices=["wikitext2", "ptb", "c4"],
    )
    return parser


def validate_args(args):
    if not 0 <= args.sparsity < 1:
        raise ValueError("--sparsity must be in [0, 1)")
    if (args.prunen == 0) != (args.prunem == 0):
        raise ValueError("--prunen and --prunem must be specified together")
    if args.prunen and not 0 < args.prunen < args.prunem:
        raise ValueError("N:M pruning requires 0 < N < M")
    if args.nsamples <= 0 or args.blocksize <= 0:
        raise ValueError("--nsamples and --blocksize must be positive")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    model, architecture = load_model(args.model, args.seqlen)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=args.model, tokenizer=tokenizer,
    )

    if args.sparsity or args.prunen:
        start = time.time()
        prune_model(model, architecture, dataloader, args)
        print(f"Pruning completed in {time.time() - start:.1f}s")
    report_sparsity(model, architecture)

    for dataset in args.eval_datasets:
        _, test_encoding = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen,
            model=args.model, tokenizer=tokenizer,
        )
        print(f"{dataset} perplexity: {evaluate(model, architecture, test_encoding):.3f}")

    if args.save:
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)


if __name__ == "__main__":
    main()
