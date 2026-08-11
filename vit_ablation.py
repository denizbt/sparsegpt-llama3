"""Run a ViT SparseGPT calibration-size ablation and write results to CSV."""

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoImageProcessor

from llmutils import report_sparsity, validate_args
from vit import VIT_ARCHITECTURE, load_vit
from vitdata import get_imagenet_loaders
from vitutils import evaluate_classifier, prune_vit


def run_ablation(
    model_factory, calibration_loader, validation_loader, sizes, args, processor=None
):
    rows = []
    dense_model = model_factory().eval()
    dense_metrics = evaluate_classifier(dense_model, validation_loader, device=args.device)
    del dense_model
    rows.append({
        "calibration_size": 0,
        "sparsity": 0.0,
        **dense_metrics,
        "top1_drop": 0.0,
        "top5_drop": 0.0,
        "pruning_seconds": 0.0,
    })

    for size in sizes:
        print(f"Calibration-size ablation: {size} images")
        model = model_factory().eval()
        prune_args = SimpleNamespace(**vars(args))
        prune_args.nsamples = size
        start = time.time()
        prune_vit(
            model, VIT_ARCHITECTURE, calibration_loader, prune_args,
            device=args.device,
        )
        elapsed = time.time() - start
        sparsity = report_sparsity(model, VIT_ARCHITECTURE)
        metrics = evaluate_classifier(model, validation_loader, device=args.device)
        rows.append({
            "calibration_size": size,
            "sparsity": sparsity,
            **metrics,
            "top1_drop": dense_metrics["top1"] - metrics["top1"],
            "top5_drop": dense_metrics["top5"] - metrics["top5"],
            "pruning_seconds": elapsed,
        })
        if args.save_dir:
            output_dir = Path(args.save_dir) / f"calibration-{size}"
            model.save_pretrained(output_dir)
            if processor is not None:
                processor.save_pretrained(output_dir)
        del model
    return rows


def write_csv(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("imagenet_root", nargs="?")
    parser.add_argument("--cached-imagenet", action="store_true")
    parser.add_argument("--calibration-sizes", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--output", default="vit-calibration-ablation.csv")
    parser.add_argument("--save-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--dataset-cache")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--wbits", type=int, default=16)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--prune-only", default="")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--true-sequential", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    sizes = sorted(set(args.calibration_sizes))
    if not sizes or sizes[0] <= 0:
        raise ValueError("Calibration sizes must be positive")
    args.nsamples = sizes[-1]
    validate_args(args)
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")

    from modelutils import DEV
    args.device = DEV if args.device == "auto" else torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model)
    calibration_loader, validation_loader = get_imagenet_loaders(
        args.imagenet_root,
        processor,
        nsamples=sizes[-1],
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
        train_split=args.train_split,
        validation_split=args.validation_split,
        cache_dir=args.dataset_cache,
        eval_samples=args.eval_samples,
        use_cached_imagenet=args.cached_imagenet,
        require_validation=True,
        eval_seed=args.seed,
    )
    rows = run_ablation(
        lambda: load_vit(args.model), calibration_loader, validation_loader, sizes, args,
        processor=processor,
    )
    write_csv(rows, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
