"""SparseGPT pruning and perplexity evaluation for GPT-2 models."""

from llmutils import Architecture, run_cli


GPT2_ARCHITECTURE = Architecture(
    block_path="transformer.h",
    input_paths=("transformer.wte", "transformer.wpe"),
    final_norm_path="transformer.ln_f",
    head_path="lm_head",
    sequential_groups=(
        ("attn.c_attn",),
        ("attn.c_proj",),
        ("mlp.c_fc",),
        ("mlp.c_proj",),
    ),
)


if __name__ == "__main__":
    run_cli(GPT2_ARCHITECTURE, "gpt2", __doc__)
