"""SparseGPT pruning and ImageNet evaluation for ViT classifiers."""

import argparse
import time

import torch
from transformers import AutoConfig, AutoImageProcessor, ViTForImageClassification

from llmutils import Architecture, report_sparsity, validate_args
from modelutils import DEV
from vitdata import get_imagenet_loaders
from vitutils import evaluate_classifier, prune_vit


VIT_ARCHITECTURE = Architecture(
    block_path="vit.encoder.layer",
    input_paths=("vit.embeddings",),
    final_norm_path="vit.layernorm",
    head_path="classifier",
    sequential_groups=(
        (
            "attention.attention.query",
            "attention.attention.key",
            "attention.attention.value",
        ),
        ("attention.output.dense",),
        ("intermediate.dense",),
        ("output.dense",),
    ),
)


def load_vit(model_name):
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type != "vit":
        raise ValueError(
            f"Expected a 'vit' checkpoint, got config.model_type={config.model_type!r}"
        )
    return ViTForImageClassification.from_pretrained(model_name)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face ViT checkpoint or local path")
    parser.add_argument(
        "imagenet_root", nargs="?", help="Directory containing train/validation folders"
    )
    parser.add_argument(
        "--cached-imagenet", action="store_true",
        help="Use a complete, previously downloaded Hugging Face ImageNet cache",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calibration-size", "--nsamples", dest="nsamples", type=int, default=512,
        help="Number of calibration images (default: 512)",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
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
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--dataset-cache")
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--save", default="")
    return parser


def main():
    args = build_parser().parse_args()
    validate_args(args)
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    model = load_vit(args.model).eval()
    device = DEV if args.device == "auto" else torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model)
    calibration_loader, validation_loader = get_imagenet_loaders(
        args.imagenet_root,
        processor,
        nsamples=args.nsamples,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
        train_split=args.train_split,
        validation_split=args.validation_split,
        cache_dir=args.dataset_cache,
        eval_samples=args.eval_samples,
        use_cached_imagenet=args.cached_imagenet,
        require_validation=not args.skip_eval,
        eval_seed=args.seed,
    )
    if args.sparsity or args.prunen:
        start = time.time()
        prune_vit(model, VIT_ARCHITECTURE, calibration_loader, args, device=device)
        print(f"Pruning completed in {time.time() - start:.1f}s")
    report_sparsity(model, VIT_ARCHITECTURE)
    if not args.skip_eval:
        metrics = evaluate_classifier(model, validation_loader, device=device)
        print(
            f"ImageNet loss: {metrics['loss']:.4f}; "
            f"top-1: {metrics['top1']:.4%}; top-5: {metrics['top5']:.4%}"
        )
    if args.save:
        model.save_pretrained(args.save)
        processor.save_pretrained(args.save)


if __name__ == "__main__":
    main()
