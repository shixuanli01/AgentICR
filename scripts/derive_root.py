"""Create a derived run root that reuses an existing phase 1.

Supplementary runs (another receiver policy, another condition, another
revision-seed replication) must reuse the exact phase-1 beliefs and StateBridge
payloads of a parent run, but write their revisions and their config updates
to a separate directory so the parent run is never modified.

The derived root gets:
  prebeliefs/  -> symlink to the parent's prebeliefs/
  messages/    -> symlink to the parent's messages/ (StateBridge payloads)
  config.json  -> copy of the parent's config, with the run-progress fields
                  cleared and, optionally, a new replication id

Changing --replication-id changes only the revision sampling seeds; the
beliefs and messages stay those of the parent.

Usage:
    python scripts/derive_root.py PARENT_ROOT NEW_ROOT [--replication-id seed_pair_01]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROGRESS_FIELDS = (
    "completed_revision_conditions",
    "revision_conditions_requested",
    "pair_class_scope",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("parent", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--replication-id")
    cli = parser.parse_args()

    config = json.loads((cli.parent / "config.json").read_text(encoding="utf-8"))
    if not (cli.parent / "prebeliefs" / "merged.jsonl").is_file():
        raise SystemExit(f"{cli.parent} has no merged phase 1 (prebeliefs/merged.jsonl)")
    cli.target.mkdir(parents=True, exist_ok=True)
    (cli.target / "logs").mkdir(exist_ok=True)
    for name in ("prebeliefs", "messages"):
        link = cli.target / name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(os.path.relpath((cli.parent / name).resolve(), cli.target.resolve()))
    for field in PROGRESS_FIELDS:
        config.pop(field, None)
    if cli.replication_id:
        config["replication_id"] = cli.replication_id
    (cli.target / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(f"derived {cli.target} from {cli.parent} (replication_id={config.get('replication_id')})")


if __name__ == "__main__":
    main()
