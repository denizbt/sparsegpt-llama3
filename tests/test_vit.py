from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from llmutils import report_sparsity
from vit import VIT_ARCHITECTURE, build_parser
from vit_ablation import run_ablation, write_csv
from vitdata import get_imagenet_loaders
from vitutils import capture_image_inputs, evaluate_classifier, prune_vit


def tiny_vit():
    config = ViTConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
        num_channels=3,
        num_labels=10,
    )
    return ViTForImageClassification(config)


def image_loader(count=8, batch_size=1):
    images = torch.randn(count, 3, 32, 32)
    labels = torch.arange(count) % 10
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def pruning_args():
    return SimpleNamespace(
        nsamples=4,
        true_sequential=False,
        minlayer=-1,
        maxlayer=1000,
        prune_only="",
        invert=False,
        wbits=16,
        sparsity=0.5,
        prunen=0,
        prunem=0,
        percdamp=0.01,
        blocksize=8,
    )


def test_vit_architecture_selects_encoder_only():
    assert VIT_ARCHITECTURE.block_path == "vit.encoder.layer"
    assert VIT_ARCHITECTURE.input_paths == ("vit.embeddings",)
    assert VIT_ARCHITECTURE.head_path == "classifier"


def test_vit_calibration_size_default_and_aliases():
    assert build_parser().parse_args(["model", "images"]).nsamples == 4096
    assert build_parser().parse_args(
        ["model", "images", "--calibration-size", "128"]
    ).nsamples == 128
    assert build_parser().parse_args(
        ["model", "images", "--nsamples", "64"]
    ).nsamples == 64
    assert build_parser().parse_args(
        ["model", "--cached-imagenet"]
    ).cached_imagenet


def test_vit_capture_accepts_batched_images():
    model = tiny_vit().eval()
    inputs, positional, keywords = capture_image_inputs(
        model, VIT_ARCHITECTURE, image_loader(count=4, batch_size=2), 3,
        torch.device("cpu"),
    )
    assert inputs.shape == (3, 17, 16)
    assert isinstance(positional, tuple)
    assert isinstance(keywords, dict)


def test_vit_sparsegpt_prunes_only_encoder_linears():
    torch.manual_seed(0)
    model = tiny_vit().eval()
    patch_before = model.vit.embeddings.patch_embeddings.projection.weight.detach().clone()
    head_before = model.classifier.weight.detach().clone()
    args = pruning_args()

    prune_vit(
        model, VIT_ARCHITECTURE, image_loader(count=args.nsamples), args,
        device=torch.device("cpu"),
    )
    sparsity = report_sparsity(model, VIT_ARCHITECTURE)
    logits = model(pixel_values=torch.randn(2, 3, 32, 32)).logits

    assert sparsity == pytest.approx(0.5, abs=0.02)
    assert torch.equal(model.vit.embeddings.patch_embeddings.projection.weight, patch_before)
    assert torch.equal(model.classifier.weight, head_before)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_vit_evaluation_metrics():
    metrics = evaluate_classifier(
        tiny_vit().eval(), image_loader(count=8, batch_size=4), torch.device("cpu")
    )
    assert set(metrics) == {"loss", "top1", "top5"}
    assert metrics["loss"] > 0
    assert 0 <= metrics["top1"] <= metrics["top5"] <= 1


def test_local_imagenet_calibration_loader(tmp_path):
    for split in ("train", "validation"):
        for class_name, color in (("class-a", 32), ("class-b", 224)):
            directory = tmp_path / split / class_name
            directory.mkdir(parents=True)
            Image.new("RGB", (32, 32), (color, color, color)).save(directory / "sample.png")
    processor = ViTImageProcessor(size={"height": 32, "width": 32})
    calibration, validation = get_imagenet_loaders(
        str(tmp_path), processor, nsamples=2, seed=0, eval_batch_size=2,
        cache_dir=str(tmp_path.parent / f"{tmp_path.name}-cache"),
    )
    calibration_images, calibration_labels = next(iter(calibration))
    validation_images, validation_labels = next(iter(validation))
    assert calibration_images.shape == (1, 3, 32, 32)
    assert calibration_labels.shape == (1,)
    assert validation_images.shape == (2, 3, 32, 32)
    assert validation_labels.shape == (2,)


def test_calibration_only_loader_does_not_require_validation(tmp_path):
    directory = tmp_path / "train" / "class-a"
    directory.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(directory / "sample.png")
    processor = ViTImageProcessor(size={"height": 32, "width": 32})
    calibration, validation = get_imagenet_loaders(
        str(tmp_path), processor, nsamples=1, require_validation=False,
        cache_dir=str(tmp_path.parent / f"{tmp_path.name}-cache"),
    )
    assert len(calibration.dataset) == 1
    assert validation is None


def test_imagenet_loader_requires_one_source():
    processor = ViTImageProcessor(size={"height": 32, "width": 32})
    with pytest.raises(ValueError, match="either imagenet_root"):
        get_imagenet_loaders(None, processor)
    with pytest.raises(ValueError, match="not both"):
        get_imagenet_loaders("local", processor, use_cached_imagenet=True)


def test_tiny_vit_calibration_size_ablation(tmp_path):
    torch.manual_seed(7)
    reference = tiny_vit().state_dict()

    def model_factory():
        model = tiny_vit()
        model.load_state_dict(reference)
        return model

    args = pruning_args()
    args.device = torch.device("cpu")
    args.save_dir = None
    rows = run_ablation(
        model_factory,
        image_loader(count=4),
        image_loader(count=8, batch_size=4),
        [2, 4],
        args,
    )
    output = tmp_path / "ablation.csv"
    write_csv(rows, output)

    assert [row["calibration_size"] for row in rows] == [0, 2, 4]
    assert rows[0]["sparsity"] == 0
    assert rows[1]["sparsity"] == pytest.approx(0.5, abs=0.02)
    assert rows[2]["sparsity"] == pytest.approx(0.5, abs=0.02)
    assert output.read_text().startswith("calibration_size,sparsity,loss,top1,top5")
