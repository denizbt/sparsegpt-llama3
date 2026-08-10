from types import SimpleNamespace

import pytest
import torch
import transformers

from gpt2 import GPT2_ARCHITECTURE
from llmutils import (
    capture_inputs,
    infer_seqlen,
    prune_model,
    report_sparsity,
    validate_args,
)
from mistral import MISTRAL_ARCHITECTURE
from qwen import QWEN_ARCHITECTURE


def pruning_args(nsamples=4):
    return SimpleNamespace(
        nsamples=nsamples,
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


def random_batches(vocab_size, seqlen, count):
    return [
        (torch.randint(0, vocab_size, (1, seqlen)), None)
        for _ in range(count)
    ]


def tiny_gpt2():
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        n_positions=16,
        vocab_size=32,
        use_cache=False,
    )
    return transformers.GPT2LMHeadModel(config)


def tiny_mistral():
    config = transformers.MistralConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        vocab_size=64,
        use_cache=False,
    )
    return transformers.MistralForCausalLM(config)


def tiny_qwen25():
    # Transformers exposes Qwen2.5 through its Qwen2 implementation.
    config = transformers.Qwen2Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        vocab_size=64,
        use_cache=False,
    )
    return transformers.Qwen2ForCausalLM(config)


MODEL_CASES = (
    pytest.param(tiny_gpt2, GPT2_ARCHITECTURE, 32, id="gpt2"),
    pytest.param(tiny_mistral, MISTRAL_ARCHITECTURE, 64, id="mistral"),
    pytest.param(tiny_qwen25, QWEN_ARCHITECTURE, 64, id="qwen2.5"),
)


def test_model_specific_block_paths():
    assert GPT2_ARCHITECTURE.block_path == "transformer.h"
    assert MISTRAL_ARCHITECTURE.block_path == "model.layers"
    assert QWEN_ARCHITECTURE.block_path == "model.layers"


def test_sequence_length_defaults_to_calibration_length():
    assert infer_seqlen(SimpleNamespace(max_position_embeddings=8192)) == 2048
    assert infer_seqlen(SimpleNamespace(n_positions=1024)) == 1024
    assert infer_seqlen(SimpleNamespace(), requested=256) == 256


@pytest.mark.parametrize(
    "sparsity,prunen,prunem,valid",
    [(0.5, 0, 0, True), (0.0, 2, 4, True), (1.0, 0, 0, False), (0.5, 2, 0, False)],
)
def test_pruning_argument_validation(sparsity, prunen, prunem, valid):
    args = pruning_args()
    args.sparsity, args.prunen, args.prunem = sparsity, prunen, prunem
    if valid:
        validate_args(args)
    else:
        with pytest.raises(ValueError):
            validate_args(args)


@pytest.mark.parametrize("factory,architecture,vocab_size", MODEL_CASES)
def test_adapter_captures_inputs(factory, architecture, vocab_size):
    model = factory().eval()
    model.seqlen = 8
    inputs, positional, keywords = capture_inputs(
        model,
        architecture,
        random_batches(vocab_size, model.seqlen, 2),
        2,
        torch.device("cpu"),
    )
    assert inputs.shape == (2, 8, model.config.hidden_size)
    assert isinstance(positional, tuple)
    assert isinstance(keywords, dict)


@pytest.mark.parametrize("factory,architecture,vocab_size", MODEL_CASES)
def test_sparsegpt_prunes_tiny_model(factory, architecture, vocab_size):
    torch.manual_seed(0)
    args = pruning_args()
    model = factory().eval()
    model.seqlen = 8
    batches = random_batches(vocab_size, model.seqlen, args.nsamples)

    with torch.no_grad():
        dense_logits = model(batches[0][0]).logits
    prune_model(model, architecture, batches, args, device=torch.device("cpu"))
    sparsity = report_sparsity(model, architecture)
    with torch.no_grad():
        sparse_logits = model(batches[0][0]).logits

    assert sparsity == pytest.approx(0.5, abs=0.02)
    assert sparse_logits.shape == dense_logits.shape
    assert torch.isfinite(sparse_logits).all()


def test_gpt2_two_of_four_pattern():
    torch.manual_seed(0)
    args = pruning_args()
    args.sparsity = 0.0
    args.prunen = 2
    args.prunem = 4
    model = tiny_gpt2().eval()
    model.seqlen = 8
    batches = random_batches(32, model.seqlen, args.nsamples)

    prune_model(model, GPT2_ARCHITECTURE, batches, args, device=torch.device("cpu"))

    # GPT-2 Conv1D weights are stored [input, output], while SparseGPT applies
    # N:M groups along the input dimension after transposing them.
    weight = model.transformer.h[0].attn.c_attn.weight.T
    groups = weight.reshape(weight.shape[0], -1, 4)
    assert torch.all(torch.count_nonzero(groups == 0, dim=2) == 2)


@pytest.mark.parametrize("factory,architecture,vocab_size", MODEL_CASES)
def test_sparse_weights_survive_save_and_reload(factory, architecture, vocab_size, tmp_path):
    torch.manual_seed(0)
    args = pruning_args(nsamples=2)
    model = factory().eval()
    model.seqlen = 8
    prune_model(
        model,
        architecture,
        random_batches(vocab_size, model.seqlen, args.nsamples),
        args,
        device=torch.device("cpu"),
    )
    expected = report_sparsity(model, architecture)
    model.save_pretrained(tmp_path)

    reloaded = type(model).from_pretrained(tmp_path)
    actual = report_sparsity(reloaded, architecture)
    assert actual == expected
