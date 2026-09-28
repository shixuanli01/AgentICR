"""Sharded Phase-2 paired belief-revision evaluation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from . import AGENTS, DIRECTIONS, direction_agents
from .channels import make_channel
from .merge import merge_prebeliefs
from .prebeliefs import accepted_fingerprints, rank_and_world
from .protocol import (
    atomic_write_json,
    atomic_write_jsonl,
    answer_is_correct,
    classify_pair,
    other_item_id,
    parse_conditions,
    revision_seed,
    sha256_json,
    sha256_text,
)
from .runtime import ICRRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICR phase 2 revisions")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        default="none,true_text,true_statebridge",
        help="Comma-separated communication conditions",
    )
    parser.add_argument(
        "--latent-steps", type=int, default=10,
        help="Latent autoregressive steps for LatentMAS conditions",
    )
    parser.add_argument(
        "--item-ids", nargs="+", type=int,
        help="Optional selected-item subset (used by resumable smoke runs)",
    )
    parser.add_argument(
        "--skip-all-correct-items", action="store_true",
        help=(
            "Skip items every agent answered correctly. Communication has "
            "almost no room to change them, and with three agents a mixed item "
            "still contains two both-correct pairs, so SCR stays measured "
            "without paying for the items where nothing can happen."
        ),
    )
    parser.add_argument(
        "--all-correct-sample", type=int, default=0,
        help=(
            "Of the items --skip-all-correct-items would drop, keep one in N "
            "(0 drops them all, 1 keeps them all). Full-set accuracy needs the "
            "fraction of skipped records the receiver still gets right, and "
            "that rate cannot be borrowed from the mixed items: those are the "
            "harder ones, so borrowing biases full-set accuracy downward. A "
            "deterministic item_id %% N subsample measures it directly."
        ),
    )
    parser.add_argument(
        "--both-correct-sample", type=int, default=1,
        help=(
            "Keep one in N both-correct items (1 keeps all). Communication has "
            "almost no room to change an item both agents already answered "
            "correctly, so a deterministic item_id %% N subsample keeps SCR "
            "measured instead of assumed while removing most of the cost."
        ),
    )
    parser.add_argument(
        "--pair-classes", type=str, default=None,
        help=(
            "Comma-separated pair classifications to run, e.g. "
            "'correction_opportunity,destruction_risk'. Supplementary conditions "
            "can be measured on the mixed-correctness directions alone, at a "
            "third of the cost; SR, SCR and retained accuracy are then NOT "
            "measured for them and must not be filled in by inference."
        ),
    )
    parser.add_argument(
        "--receiver-policy", choices=("revise", "verify"), default="revise",
        help=(
            "How the receiver integrates the message. 'revise' is the frozen V3 "
            "instruction; 'verify' is the V4 policy that checks the message "
            "against the problem before deciding. Changing it changes the "
            "protocol, so every compared condition must use the same value."
        ),
    )
    parser.add_argument(
        "--global-resume", action="store_true",
        help="Resume completed keys from every revision shard, enabling safe resharding",
    )
    parser.add_argument(
        "--rerun-truncated", action="store_true",
        help="Regenerate cached records that never reached EOS; see icr.prebeliefs.",
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional tag separating concurrent condition-group shard outputs",
    )
    return parser.parse_args()


def load_prebelief_map(root: Path) -> dict[tuple[int, str], dict[str, Any]]:
    merged = root / "prebeliefs" / "merged.jsonl"
    if not merged.exists():
        merge_prebeliefs(root, require_complete=True)
    records = [json.loads(line) for line in merged.read_text(encoding="utf-8").split("\n") if line]
    return {(int(row["item_id"]), str(row["agent_id"])): row for row in records}


def main() -> None:
    cli = parse_args()
    rank, world = rank_and_world(cli)
    conditions = parse_conditions(cli.conditions)
    config_path = cli.artifact_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prebeliefs = load_prebelief_map(cli.artifact_root)
    selected_ids = [int(value) for value in config["selected_item_ids"]]
    expected_prebeliefs = {
        (item_id, agent_id) for item_id in selected_ids for agent_id in AGENTS
    }
    if set(prebeliefs) != expected_prebeliefs:
        raise RuntimeError("Phase-1 cache is incomplete or contains unexpected records")

    # Text and StateBridge condition groups can start concurrently. Serialize
    # their manifest updates so neither group can lose the other's condition
    # list through a read/modify/write race.
    lock_path = cli.artifact_root / ".config.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        existing_conditions = list(config.get("revision_conditions_requested", []))
        combined_conditions = list(dict.fromkeys((*existing_conditions, *conditions)))
        config["revision_conditions_requested"] = combined_conditions
        config["receiver_policy"] = str(cli.receiver_policy)
        if cli.pair_classes:
            config.setdefault("pair_class_scope", {})[
                ",".join(sorted(cli.conditions.split(",")))
            ] = sorted({p.strip() for p in cli.pair_classes.split(",") if p.strip()})
        config["revision_prompt_version"] = (
            "icr_v4_verify_then_decide" if cli.receiver_policy == "verify"
            else config.get("revision_prompt_version", "icr_v3_mid_injection")
        )
        config["both_correct_sampling"] = {
            "keep_one_in": int(cli.both_correct_sample),
            "rule": "item_id % keep_one_in == 0",
            "skip_all_correct_items": bool(cli.skip_all_correct_items),
            "all_correct_keep_one_in": int(cli.all_correct_sample),
            "label_free": True,
        }
        atomic_write_json(config_path, config)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    run_ids = selected_ids
    if cli.item_ids is not None:
        unknown_ids = sorted(set(cli.item_ids) - set(selected_ids))
        if unknown_ids:
            raise ValueError(f"Requested item IDs are not in this run: {unknown_ids}")
        run_ids = [item_id for item_id in selected_ids if item_id in set(cli.item_ids)]
    directional_pairs = [
        (item_id, direction)
        for item_id in run_ids
        for direction in DIRECTIONS
    ]
    # Shard over the pairs that still need work, not over every pair. The
    # records a dead worker left behind are exactly one stride of the old world
    # size, and a stride nests inside any divisor of it -- pairs[7::8] sits
    # entirely inside pairs[3::4] and pairs[1::2] -- so re-running at a smaller
    # world hands the whole backlog to a single worker again. ARC-Challenge
    # filled 293 records that way on one GPU while three sat idle.
    if cli.global_resume:
        done: set[tuple[int, str, str]] = set()
        for existing_path in cli.artifact_root.glob(
            "revisions/rank*/records/item_*.json"
        ):
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if existing.get("status") == "complete":
                done.add(
                    (
                        int(existing["item_id"]),
                        str(existing["direction"]),
                        str(existing["condition"]),
                    )
                )
        outstanding = [
            pair
            for pair in directional_pairs
            if any((pair[0], pair[1], condition) not in done for condition in conditions)
        ]
        assigned = outstanding[rank::world]
    else:
        assigned = directional_pairs[rank::world]
    needs_other_source = any(condition.startswith("other_") for condition in conditions)
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"[ICR revisions rank={rank}] stop requested signal={signum}", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    device_index = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    runtime = ICRRuntime(
        model_name=config["model"],
        device=torch.device(f"cuda:{device_index}"),
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        task=str(config.get("dataset", "medqa")),
    )
    suffix = f"_{cli.output_tag}" if cli.output_tag else ""
    shard_name = f"rank{rank}{suffix}"
    runtime.receiver_policy = str(cli.receiver_policy)
    wanted_pair_classes = (
        {p.strip() for p in cli.pair_classes.split(",") if p.strip()}
        if cli.pair_classes else None
    )
    records_dir = cli.artifact_root / "revisions" / shard_name / "records"
    globally_completed: set[tuple[int, str, str]] = set()
    truncated_keys: set[tuple[int, str, str]] = set()
    if cli.global_resume:
        for existing_path in cli.artifact_root.glob(
            "revisions/rank*/records/item_*.json"
        ):
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if existing.get("status") == "complete" and existing.get(
                "config_fingerprint"
            ) in accepted_fingerprints(config):
                key = (
                    int(existing["item_id"]),
                    str(existing["direction"]),
                    str(existing["condition"]),
                )
                globally_completed.add(key)
                if cli.rerun_truncated and not bool(existing.get("hit_eos", True)):
                    truncated_keys.add(key)
    print(
        f"[ICR revisions rank={rank}] pairs={len(assigned)} conditions={conditions}",
        flush=True,
    )
    for item_id, direction in assigned:
        sender_id, receiver_id = direction_agents(direction)
        sender = prebeliefs[(item_id, sender_id)]
        receiver = prebeliefs[(item_id, receiver_id)]
        other_id = None
        other_sender = None
        if needs_other_source:
            other_id = other_item_id(
                item_id, selected_ids, int(config["other_mapping_offset"])
            )
            other_sender = prebeliefs[(other_id, sender_id)]
        seed = revision_seed(
            int(config["global_seed"]),
            item_id,
            direction,
            config.get("replication_id"),
        )
        pair_class = classify_pair(bool(sender["correct"]), bool(receiver["correct"]))
        if wanted_pair_classes is not None and pair_class not in wanted_pair_classes:
            continue
        if cli.skip_all_correct_items and all(
            bool(prebeliefs[(item_id, agent)]["correct"]) for agent in AGENTS
        ):
            sampled = (
                cli.all_correct_sample > 0
                and item_id % cli.all_correct_sample == 0
            )
            if not sampled:
                continue
        if (
            cli.both_correct_sample > 1
            and pair_class == "both_correct"
            and item_id % cli.both_correct_sample != 0
        ):
            continue

        for condition in conditions:
            record_path = records_dir / f"item_{item_id:04d}_{direction}_{condition}.json"
            record_key = (item_id, direction, condition)
            if (
                cli.global_resume
                and record_key in globally_completed
                and record_key not in truncated_keys
            ):
                print(
                    f"[ICR revisions rank={rank}] global-resume item={item_id} "
                    f"direction={direction} condition={condition}",
                    flush=True,
                )
                continue
            if record_path.exists():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                truncated = not bool(cached.get("hit_eos", True))
                if (
                    cached.get("status") == "complete"
                    and cached.get("config_fingerprint") in accepted_fingerprints(config)
                    and int(cached.get("revision_seed", -1)) == seed
                    and not (cli.rerun_truncated and truncated)
                ):
                    print(
                        f"[ICR revisions rank={rank}] resume item={item_id} "
                        f"direction={direction} condition={condition}",
                        flush=True,
                    )
                    continue
            channel = make_channel(condition)
            message = channel.build_message(
                sender,
                receiver,
                {
                    "artifact_root": cli.artifact_root,
                    "tokenizer": runtime.model.tokenizer,
                    "other_sender_record": other_sender,
                },
            )
            if condition.endswith("_latentmas"):
                generated = runtime.generate_latentmas_revision(
                    question=str(receiver["question"]),
                    receiver_reasoning=str(receiver["reasoning_text"]),
                    receiver_answer=receiver["parsed_answer"],
                    message=message,
                    seed=seed,
                    latent_steps=cli.latent_steps,
                )
                message.diagnostics.update(generated["latentmas_compute"])
                reference_path = (
                    cli.artifact_root
                    / "messages"
                    / "latentmas"
                    / f"item_{item_id:04d}_{direction}_{condition}.json"
                )
                atomic_write_json(
                    reference_path,
                    {
                        "format": "icr_latentmas_reconstructible_reference_v1",
                        "condition": condition,
                        "source_item_id": message.source_item_id,
                        "source_agent_id": message.source_agent_id,
                        "source_prebelief_index": {
                            "file": "prebeliefs/merged.jsonl",
                            "item_id": message.source_item_id,
                            "agent_id": message.source_agent_id,
                        },
                        "source_prompt_sha256": message.trajectory["prompt_sha256"],
                        "source_generated_token_ids_sha256": sha256_json(
                            message.trajectory["generated_token_ids"]
                        ),
                        "algorithm": {
                            "upstream_commit": "9a9e4d331eb11430bd9e64754c6b252b06d73031",
                            "all_layers": True,
                            "all_source_positions": True,
                            "latent_steps": cli.latent_steps,
                            "latent_space_realign": False,
                        },
                        "measured": message.diagnostics,
                    },
                )
                generated["latentmas_message_reference_file"] = str(
                    reference_path.relative_to(cli.artifact_root)
                )
            else:
                generated = runtime.generate_revision(
                    question=str(receiver["question"]),
                    receiver_reasoning=str(receiver["reasoning_text"]),
                    receiver_answer=receiver["parsed_answer"],
                    message=message,
                    seed=seed,
                )
            post_answer = generated["parsed_answer"]
            gold = receiver["gold"]
            record = {
                "status": "complete",
                "config_fingerprint": config["fingerprint"],
                "item_id": item_id,
                "direction": direction,
                "condition": condition,
                "replication_id": config.get("replication_id") or "seed_pair_00",
                "benchmark": config.get("benchmark", "medqa300"),
                "task": config.get("dataset", "medqa"),
                "sender_agent_id": sender_id,
                "receiver_agent_id": receiver_id,
                "sender_pre_answer": sender["parsed_answer"],
                "sender_correct": bool(sender["correct"]),
                "receiver_pre_answer": receiver["parsed_answer"],
                "receiver_pre_correct": bool(receiver["correct"]),
                "receiver_prior_sha256": sha256_text(str(receiver["reasoning_text"])),
                "gold": gold,
                "pair_classification": pair_class,
                "other_item_id": other_id,
                "message_source_item_id": message.source_item_id,
                "message_source_agent_id": message.source_agent_id,
                "communication_payload": message.diagnostics,
                **generated,
                "receiver_post_answer": post_answer,
                "receiver_post_correct": answer_is_correct(
                    str(config.get("dataset", "medqa")), post_answer, gold
                ),
                "answer_changed": post_answer != receiver["parsed_answer"],
                "followed_sender": post_answer == sender["parsed_answer"],
                "completed_at": datetime.now().isoformat(),
            }
            atomic_write_json(record_path, record)
            globally_completed.add(record_key)
            print(
                f"[ICR revisions rank={rank}] item={item_id} direction={direction} "
                f"condition={condition} pre={receiver['parsed_answer']} post={post_answer} "
                f"gold={gold} correct={record['receiver_post_correct']}",
                flush=True,
            )
            del message
            if stop_requested:
                break
        if stop_requested:
            break
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(records_dir.glob("item_*.json"))
    ]
    atomic_write_jsonl(cli.artifact_root / "revisions" / f"{shard_name}.jsonl", rows)


if __name__ == "__main__":
    main()
