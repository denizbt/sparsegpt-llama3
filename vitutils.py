"""Vision-specific activation collection, pruning, and evaluation helpers."""

import torch
import torch.nn as nn

from llmutils import move_optional, resolve_attr, selected
from modelutils import DEV, empty_cache, find_layers
from quant import Quantizer
from sparsegpt import SparseGPT


@torch.no_grad()
def capture_image_inputs(model, architecture, dataloader, nsamples, device):
    blocks = resolve_attr(model, architecture.block_path)
    moved_paths = move_optional(model, architecture.input_paths, device)
    blocks[0] = blocks[0].to(device)
    captured = []
    captured_count = 0
    cache = {"args": (), "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, *args, **kwargs):
            nonlocal captured_count
            remaining = nsamples - captured_count
            captured.append(hidden_states[:remaining].detach())
            captured_count += min(hidden_states.shape[0], remaining)
            cache["args"] = args
            cache["kwargs"] = kwargs
            raise StopIteration

    blocks[0] = Catcher(blocks[0])
    try:
        for pixel_values, _ in dataloader:
            if captured_count >= nsamples:
                break
            try:
                model(pixel_values=pixel_values.to(device))
            except StopIteration:
                pass
    finally:
        blocks[0] = blocks[0].module
        blocks[0] = blocks[0].cpu()
        move_optional(model, moved_paths, torch.device("cpu"))
        empty_cache(device)
    if captured_count != nsamples:
        raise RuntimeError(f"Captured {captured_count} images, expected {nsamples}")
    if cache["kwargs"] is None:
        raise RuntimeError("The first ViT encoder block did not receive any inputs")
    return torch.cat(captured, dim=0), cache["args"], cache["kwargs"]


@torch.no_grad()
def prune_vit(model, architecture, dataloader, args, device=DEV):
    blocks = resolve_attr(model, architecture.block_path)
    inputs, block_args, block_kwargs = capture_image_inputs(
        model, architecture, dataloader, args.nsamples, device
    )
    outputs = torch.zeros_like(inputs)

    for block_index in range(len(blocks)):
        print(f"Block {block_index}/{len(blocks) - 1}")
        block = blocks[block_index].to(device)
        try:
            all_layers = find_layers(block)
            groups = (
                architecture.sequential_groups
                if args.true_sequential else (tuple(all_layers),)
            )
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
                try:
                    for sample in range(args.nsamples):
                        outputs[sample] = block(
                            inputs[sample].unsqueeze(0), *block_args, **block_kwargs
                        )[0]
                finally:
                    for handle in handles:
                        handle.remove()

                try:
                    for name, pruner in pruners.items():
                        print(f"Pruning block {block_index}: {name}")
                        pruner.fasterprune(
                            args.sparsity, prunen=args.prunen, prunem=args.prunem,
                            percdamp=args.percdamp, blocksize=args.blocksize,
                        )
                finally:
                    for pruner in pruners.values():
                        pruner.free()

            for sample in range(args.nsamples):
                outputs[sample] = block(
                    inputs[sample].unsqueeze(0), *block_args, **block_kwargs
                )[0]
        finally:
            blocks[block_index] = block.cpu()
            del block
            empty_cache(device)
        inputs, outputs = outputs, inputs


@torch.no_grad()
def evaluate_classifier(model, dataloader, device=DEV):
    model = model.to(device)
    total = top1 = top5 = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    try:
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=pixel_values).logits
            loss_sum += loss_fn(logits, labels).item()
            predictions = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
            top1 += (predictions[:, 0] == labels).sum().item()
            top5 += (predictions == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.numel()
    finally:
        model.cpu()
        empty_cache(device)
    if total == 0:
        raise ValueError("The validation loader is empty")
    return {"loss": loss_sum / total, "top1": top1 / total, "top5": top5 / total}
