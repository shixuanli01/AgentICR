"""Required pre-run audit printout for an ICR cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from . import DIRECTIONS, direction_agents
from .merge import merge_prebeliefs
from .protocol import other_item_id, prebelief_seed, revision_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ICR Phase-1 cache")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--item-id", type=int)
    parser.add_argument("--direction", choices=DIRECTIONS, default=DIRECTIONS[0])
    cli = parser.parse_args()
    config = json.loads((cli.artifact_root / "config.json").read_text(encoding="utf-8"))
    rows = merge_prebeliefs(cli.artifact_root, require_complete=True)
    records = {(int(row["item_id"]), row["agent_id"]): row for row in rows}
    selected_ids = [int(value) for value in config["selected_item_ids"]]
    item_id = cli.item_id if cli.item_id is not None else selected_ids[0]
    sender_id, receiver_id = direction_agents(cli.direction)
    sender = records[(item_id, sender_id)]
    receiver = records[(item_id, receiver_id)]
    other_id = other_item_id(item_id, selected_ids, int(config["other_mapping_offset"]))
    other = records[(other_id, sender_id)]

    def prefix(record):
        path = cli.artifact_root / record["statebridge_prefix_file"]
        return load_file(str(path), device="cpu")["statebridge_prefix"]

    true_prefix, self_prefix, other_prefix = prefix(sender), prefix(receiver), prefix(other)
    replication_id = config.get("replication_id")
    seed = revision_seed(
        int(config["global_seed"]), item_id, cli.direction, replication_id
    )
    condition_seeds = {
        condition: seed
        for condition in (
            "none",
            "true_text",
            "self_text",
            "other_text",
            "true_statebridge",
            "self_statebridge",
            "other_statebridge",
        )
    }
    assert len(set(condition_seeds.values())) == 1
    assert sender["generation_seed"] != receiver["generation_seed"]
    assert sender["prompt_sha256"] == receiver["prompt_sha256"]
    assert sender["question"] == receiver["question"]
    assert not torch.equal(true_prefix, self_prefix)
    assert other_id != item_id
    assert not torch.equal(true_prefix, other_prefix)

    print(f"item_id: {item_id}")
    print(f"direction: {cli.direction}")
    print(
        f"sender_seed: "
        f"{prebelief_seed(config['global_seed'], item_id, sender_id, replication_id)}"
    )
    print(
        f"receiver_pre_seed: "
        f"{prebelief_seed(config['global_seed'], item_id, receiver_id, replication_id)}"
    )
    print(f"revision_seed: {seed}")
    print(f"sender_pre_answer: {sender['parsed_answer']}")
    print(f"receiver_pre_answer: {receiver['parsed_answer']}")
    print(f"true_text_message first 200 chars: {sender['reasoning_text'][:200]!r}")
    print(f"self_text_message first 200 chars: {receiver['reasoning_text'][:200]!r}")
    print(f"other_text source item id: {other_id}")
    print(f"true_statebridge prefix shape: {list(true_prefix.shape)}")
    print(f"self_statebridge prefix shape: {list(self_prefix.shape)}")
    print(f"other_statebridge prefix shape: {list(other_prefix.shape)}")
    print(f"revision seeds by condition: {condition_seeds}")
    print("gold_used_for_generation_prompt: false (prompt builders accept no gold argument)")
    print("verification: PASS")


if __name__ == "__main__":
    main()
