"""SparseGPT pruning and perplexity evaluation for Qwen2.5 models."""

from llmutils import Architecture, run_cli


# Transformers identifies Qwen2.5 checkpoints with model_type="qwen2".
QWEN_ARCHITECTURE = Architecture(
    block_path="model.layers",
    input_paths=("model.embed_tokens", "model.rotary_emb"),
    final_norm_path="model.norm",
    head_path="lm_head",
    sequential_groups=(
        ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"),
        ("self_attn.o_proj",),
        ("mlp.up_proj", "mlp.gate_proj"),
        ("mlp.down_proj",),
    ),
)


if __name__ == "__main__":
    run_cli(QWEN_ARCHITECTURE, "qwen2", __doc__)
