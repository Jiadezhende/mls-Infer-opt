import argparse
import json
from pathlib import Path

import torch


def normal(shape: tuple[int, ...], scale: float = 0.02) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32) * scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with Path(args.config).open() as f:
        config = json.load(f)

    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_layers = int(config["num_hidden_layers"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])

    q_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim

    state: dict[str, torch.Tensor] = {}
    state["embed_tokens.weight"] = normal((vocab_size, hidden_size))

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"
        state[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size)
        state[f"{prefix}.self_attn.q_proj.weight"] = normal((q_out, hidden_size))
        state[f"{prefix}.self_attn.k_proj.weight"] = normal((kv_out, hidden_size))
        state[f"{prefix}.self_attn.v_proj.weight"] = normal((kv_out, hidden_size))
        state[f"{prefix}.self_attn.o_proj.weight"] = normal((hidden_size, q_out))
        state[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size)
        state[f"{prefix}.mlp.gate_proj.weight"] = normal((intermediate_size, hidden_size))
        state[f"{prefix}.mlp.up_proj.weight"] = normal((intermediate_size, hidden_size))
        state[f"{prefix}.mlp.down_proj.weight"] = normal((hidden_size, intermediate_size))

    state["norm.weight"] = torch.ones(hidden_size)
    state["lm_head.weight"] = normal((vocab_size, hidden_size))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
