"""Sharded Phase-1 independent-belief generation and StateBridge caching."""

from __future__ import annotations

import argparse
import json
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from . import AGENTS, CONDITIONS, PROTOCOL_V3
from .benchmarks import (
    SPECS,
    benchmark_spec,
    load_benchmark,
    select_item_ids,
    structural_exclusions,
)
from .prompts_v3 import PROMPT_VERSION
from .protocol import (
    atomic_write_json,
    atomic_write_jsonl,
    answer_is_correct,
    canonical_gold,
    prebelief_seed,
    sha256_json,
)
from .runtime import ICRRuntime, atomic_save_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{PROTOCOL_V3} phase 1")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--task", choices=sorted(SPECS), default="medqa")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--replication-id")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--item-ids", nargs="+", type=int)
    parser.add_argument(
        "--rerun-truncated", action="store_true",
        help=(
            "Regenerate cached records that never reached EOS. Pair it with a "
            "raised token budget: a truncated generation never stated its answer "
            "and was scored wrong, while a record that reached EOS is unaffected "
            "by the budget and stays valid."
        ),
    )
    parser.add_argument(
        "--only-item-ids", nargs="+", type=int,
        help=(
            "Work only on these items of this rank's shard, in the order given. "
            "It narrows which worker does the work, not what the run contains: "
            "it is not part of config.json or its fingerprint, and records land "
            "in this rank's directory with the same seeds. Used to let an idle "
            "GPU take a lagging rank's remaining items from the other end."
        ),
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    return parser.parse_args()


def rank_and_world(cli: argparse.Namespace) -> tuple[int, int]:
    rank = cli.rank if cli.rank is not None else int(os.getenv("LOCAL_RANK", "0"))
    world = cli.world_size if cli.world_size is not None else int(os.getenv("WORLD_SIZE", "1"))
    if world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid rank/world-size topology")
    return rank, world


def build_config(
    cli: argparse.Namespace,
    selected_ids: list[int],
    data: list[dict],
    excluded_ids: list[int],
) -> dict[str, Any]:
    stable = {
        "protocol": PROTOCOL_V3,
        "dataset": cli.task,
        "benchmark": benchmark_spec(cli.task).label,
        "dataset_rows": len(data),
        "selected_item_ids": selected_ids,
        "excluded_item_ids": excluded_ids,
        "exclusion_rule": (
            "arc_challenge: trailing option count != 4 (label-free)"
            if excluded_ids
            else None
        ),
        "dataset_sha256": sha256_json(data),
        "selection": {
            "mode": (
                "explicit_ids" if cli.item_ids else
                "seeded_sample" if cli.sample_size is not None else
                "prefix" if cli.limit is not None else "full"
            ),
            "sample_size": cli.sample_size,
            "selection_seed": cli.selection_seed if cli.sample_size is not None else None,
        },
        "model": cli.model,
        "global_seed": cli.global_seed,
        "generation": {
            "do_sample": True,
            "temperature": cli.temperature,
            "top_p": cli.top_p,
            "top_k": None,
            "max_new_tokens": cli.max_new_tokens,
        },
        "statebridge": {
            "selection_method": "last_k",
            "max_prefix_tokens": 64,
            "adaptive_reg": 0.001,
            "snap_ratio": 0.3,
            "prefix_strategy": "scale",
            "prefix_scale": 1.0,
            "use_hook": True,
            "algorithm_source": "methods/state_bridge.py (unmodified)",
            "receiver_injection_position": "message_slot_after_receiver_prior",
        },
        "revision_conditions": list(CONDITIONS),
        "solver_prompt_version": "icr_v2_phase1_verbatim",
        "revision_prompt_version": PROMPT_VERSION,
        "answer_parser_version": "icr.parsing_v3@ICR-V3",
        "other_mapping_offset": 137,
    }
    if cli.replication_id is not None:
        stable["replication_id"] = cli.replication_id
        stable["statebridge"]["message_directory"] = "messages/statebridge"
    return {**stable, "fingerprint": sha256_json(stable)}


# Keys that config.json gains after build_config has already hashed it. They
# record what a run has done, not what the run is, so they must stay out of the
# fingerprint. Hashing them once made every raised budget unreachable: the
# on-disk fingerprint covered them, the rebuilt candidate did not, and every
# worker refused to start.
RUNTIME_CONFIG_KEYS = (
    "fingerprint",
    "revision_conditions_requested",
    "completed_revision_conditions",
    "both_correct_sampling",
    "superseded_fingerprints",
    "max_new_tokens_history",
)


def fingerprint_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of a config that its fingerprint is computed over."""
    return {k: v for k, v in config.items() if k not in RUNTIME_CONFIG_KEYS}


def accepted_fingerprints(config: Mapping[str, Any]) -> set[str]:
    """Fingerprints whose finished records this run still trusts.

    Raising the token budget changes the fingerprint but cannot change a record
    that already reached EOS, so the superseded fingerprints stay acceptable.
    """
    return {config["fingerprint"], *config.get("superseded_fingerprints", [])}


def ensure_config(path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != candidate["fingerprint"]:
            raise RuntimeError("Artifact root contains a different ICR configuration")
        return existing
    atomic_write_json(path, candidate)
    return candidate


def main() -> None:
    cli = parse_args()
    if cli.max_new_tokens is None:
        # An existing root's own config is authoritative. SPECS only supplies
        # the budget for a root that does not exist yet, so a budget raised on
        # disk cannot drift from the value the runners rebuild their config
        # with, and raising it again needs no source edit.
        existing_config = cli.artifact_root / "config.json"
        if existing_config.is_file():
            cli.max_new_tokens = int(
                json.loads(existing_config.read_text(encoding="utf-8"))["generation"][
                    "max_new_tokens"
                ]
            )
        else:
            cli.max_new_tokens = benchmark_spec(cli.task).default_max_new_tokens
    rank, world = rank_and_world(cli)
    data = load_benchmark(cli.task)
    selected_ids = select_item_ids(
        len(data),
        item_ids=cli.item_ids,
        limit=cli.limit,
        sample_size=cli.sample_size,
        selection_seed=cli.selection_seed,
    )
    excluded_ids = structural_exclusions(cli.task, data)
    if excluded_ids:
        removed = sorted(set(selected_ids) & set(excluded_ids))
        selected_ids = [value for value in selected_ids if value not in set(excluded_ids)]
        print(
            f"[ICR prebeliefs] structural exclusion removed {len(removed)} items: {removed}",
            flush=True,
        )
    config = ensure_config(
        cli.artifact_root / "config.json",
        build_config(cli, selected_ids, data, excluded_ids),
    )
    # A repair regenerates a small, unevenly spread subset. Striding over every
    # item leaves most workers with nothing to do -- ARC-Challenge finished one
    # pass with 17 records left and only 2 of 8 workers still alive. Shard over
    # the records that actually need work instead, and write each one back to
    # the rank directory it already lives in so no duplicate appears elsewhere.
    truncated_paths: dict[tuple[int, str], Path] = {}
    if cli.rerun_truncated:
        for existing in (cli.artifact_root / "prebeliefs").glob(
            "rank*/records/item_*.json"
        ):
            try:
                row = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not bool(row.get("hit_eos", True)):
                truncated_paths[(int(row["item_id"]), str(row["agent_id"]))] = existing
        assigned_ids = sorted({item for item, _ in truncated_paths})[rank::world]
    else:
        assigned_ids = selected_ids[rank::world]
    if cli.only_item_ids:
        outside = sorted(set(cli.only_item_ids) - set(assigned_ids))
        if outside:
            raise ValueError(f"--only-item-ids outside rank {rank}'s shard: {outside}")
        assigned_ids = list(cli.only_item_ids)
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"[ICR prebeliefs rank={rank}] stop requested signal={signum}", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Shard rank and local CUDA index are deliberately decoupled so a formal
    # run can place multiple independent workers on each physical GPU.
    device_index = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    runtime = ICRRuntime(
        model_name=cli.model,
        device=torch.device(f"cuda:{device_index}"),
        max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        top_p=cli.top_p,
        task=cli.task,
    )
    records_dir = cli.artifact_root / "prebeliefs" / f"rank{rank}" / "records"
    message_directory = str(
        config["statebridge"].get("message_directory", "statebridge_messages")
    )
    prefix_dir = cli.artifact_root / message_directory / f"rank{rank}"
    print(
        f"[ICR prebeliefs rank={rank}] items={len(assigned_ids)} model={cli.model}",
        flush=True,
    )
    for item_id in assigned_ids:
        item = data[item_id]
        for agent_id in AGENTS:
            if cli.rerun_truncated and (item_id, agent_id) not in truncated_paths:
                continue
            record_path = truncated_paths.get(
                (item_id, agent_id),
                records_dir / f"item_{item_id:04d}_{agent_id}.json",
            )
            if record_path.exists():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                prefix = cli.artifact_root / cached.get("statebridge_prefix_file", "")
                truncated = not bool(cached.get("hit_eos", True))
                if (
                    cached.get("status") == "complete"
                    and cached.get("config_fingerprint") in accepted_fingerprints(config)
                    and prefix.is_file()
                    and not (cli.rerun_truncated and truncated)
                ):
                    print(f"[ICR prebeliefs rank={rank}] resume item={item_id} agent={agent_id}", flush=True)
                    continue
            seed = prebelief_seed(
                cli.global_seed, item_id, agent_id, cli.replication_id
            )
            generated = runtime.generate_prebelief(str(item["question"]), seed=seed)
            relative_prefix = Path(message_directory) / f"rank{rank}" / f"item_{item_id:04d}_{agent_id}.safetensors"
            prefix_path = cli.artifact_root / relative_prefix
            atomic_save_prefix(prefix_path, generated.pop("prefix"))
            gold = canonical_gold(cli.task, item.get("gold"))
            record = {
                "status": "complete",
                "config_fingerprint": config["fingerprint"],
                "item_id": item_id,
                "agent_id": agent_id,
                "replication_id": cli.replication_id or "seed_pair_00",
                "benchmark": config["benchmark"],
                "task": cli.task,
                "question": str(item["question"]),
                "gold": gold,
                **generated["record"],
                "correct": answer_is_correct(
                    cli.task, generated["record"]["parsed_answer"], gold
                ),
                "statebridge_prefix_file": str(relative_prefix),
                "statebridge_prefix_shape": list(generated["record"]["statebridge"]["K"] and [1, generated["record"]["statebridge"]["K"], runtime.bridge.hidden_size]),
                "statebridge_prefix_dtype": str(runtime.bridge.dtype).replace("torch.", ""),
                "completed_at": datetime.now().isoformat(),
            }
            atomic_write_json(record_path, record)
            print(
                f"[ICR prebeliefs rank={rank}] item={item_id} agent={agent_id} "
                f"pred={record['parsed_answer']} gold={gold} correct={record['correct']} "
                f"tokens={record['generation_length']}",
                flush=True,
            )
            if stop_requested:
                break
        if stop_requested:
            break
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(records_dir.glob("item_*.json"))
    ]
    atomic_write_jsonl(cli.artifact_root / "prebeliefs" / f"rank{rank}.jsonl", rows)


if __name__ == "__main__":
    main()
