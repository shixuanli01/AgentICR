# AgentICR

Code for the **Independent -> Communicate -> Revise (ICR)** evaluation of
communication channels between LLM agents.

Three agents answer each question independently. For every ordered
(sender, receiver) pair the receiver then sees a message built from the
sender's independent solution and revises its own answer once. Because the
correctness of both initial answers is known, every revision falls into one of
four strata, and the channel is scored on how it moves the receiver in each:

| Stratum | Sender | Receiver | Metric (receiver correct after revision) |
|---|---|---|---|
| correction opportunity | right | wrong | **CR** (correction rate) |
| destruction risk | wrong | right | **PR** (preservation rate) |
| both wrong | wrong | wrong | **SR** (rescue rate) |
| both right | right | right | **SCR** (retention rate) |

`SI = (CR + PR) / 2`. A receiver that never changes its answer scores SI = 50.

Communication conditions (the only thing that varies between them is the
message slot of one fixed revision prompt):

| Condition key | Name | Message |
|---|---|---|
| `none` | No Message | none |
| `true_answer` | Answer Only | the sender's parsed final answer |
| `true_text` | Full Text | the sender's post-thinking response |
| `true_statebridge` | StateBridge | up to 64 aligned hidden states of the sender |
| `true_latentmas` | LatentMAS | the sender's KV cache plus 10 latent steps |

Two receiver policies are implemented: `revise` (Critical Evaluation, the
default) and `verify` (Structured Verification).

---

## 1. Repository layout

```
icr/
  prebeliefs.py   phase 1: independent answers of agents A, B, C (+ StateBridge payloads)
  revisions.py    phase 2: one revision per (item, direction, condition)
  runtime.py      prompt rendering and generation for both phases
  channels.py     message construction for each condition
  protocol.py     phase-1 prompts, parsing/scoring entry points, seeds
  prompts_v3.py   revision prompt, Critical Evaluation (receiver policy "revise")
  prompts_v4.py   revision prompt, Structured Verification (receiver policy "verify")
  parsing_v3.py   answer extraction and numeric comparison
  benchmarks.py   benchmark specs, item selection, structural exclusions
  merge.py        merge per-worker shards into merged.jsonl
  verify.py       phase-1 cache sanity checks
  analyze.py      CR / PR / SI / SR / SCR with paired bootstrap intervals
methods/state_bridge.py   StateBridge (hidden-state selection, alignment, injection)
models.py, data.py, prompts.py, utils.py   model wrapper, dataset loaders, shared helpers
scripts/run_icr.sh         full run for one benchmark (phase 1 + phase 2 + summary)
scripts/run_icr_mixed.sh   phase 2 on mixed-correctness directions only
scripts/derive_root.py     new run root that reuses an existing phase 1
tests/                     unit tests (no GPU, no data needed)
data/README.md             how to obtain the two local evaluation files
```

---

## 2. Environment

Reported runs: Python 3.12.3, PyTorch 2.7.1 (CUDA 12.8), Transformers 4.51.3,
NVIDIA RTX 5090 (32 GB) GPUs, bfloat16, one sequence per process.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # pick the index for your CUDA
pip install -r requirements.txt
python -m pytest -q tests          # 42 tests, CPU only, should all pass
```

All commands below are run **from the repository root**; the scripts set
`PYTHONPATH` themselves. When calling a module directly, use
`PYTHONPATH=. python -m icr.<module>`.

GPU memory per worker process: about 10 GB for Qwen3-4B (two workers fit on a
32 GB GPU for the text and StateBridge conditions) and about 20 GB for
Qwen3-8B. **Run LatentMAS with one worker per GPU**; two workers per GPU run out
of memory on 32 GB.

---

## 3. Model

The reported runs use `Qwen/Qwen3-4B` at revision
`1cfa9a7208912126459214e8b04321603b3df60c` (additional runs:
`Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`). The loader
first looks for `./models/<ModelName>`, so pin the revision by downloading it
there:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --local-dir models/Qwen3-4B
# optional: huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/Qwen3-8B
```

Keep `MODEL=Qwen/Qwen3-4B` (the default); the local copy is picked up
automatically and the model id recorded in `config.json` stays the same.

---

## 4. Data

Follow `data/README.md`: download `data/medqa.json` and
`data/gpqa_diamond.json` and check their SHA-256. ARC-Challenge, GSM8K and
HumanEval+ are downloaded from Hugging Face automatically. After phase 1, compare
`dataset_sha256` in the run's `config.json` with the table in `data/README.md`.

| Task key | Benchmark | Items evaluated |
|---|---|---:|
| `medqa` | MedQA, fixed 300-question subset | 300 |
| `arc_challenge` | ARC-Challenge test | 1,165 (7 items with 3 or 5 options excluded) |
| `gsm8k` | GSM8K test | 1,319 |
| `gpqa` | GPQA-Diamond | 198 |
| `humanevalplus` | HumanEval+ | 164 |

Item IDs are 0-based row indices of the loaded benchmark. The ARC-Challenge
exclusions keep the original row indices (121, 385, 400, 836, 868, 1037, 1042).

---

## 5. Smoke test (a few minutes on one GPU)

```bash
BENCHMARK_LIMIT=4 CUDA_DEVICES="0" \
  bash scripts/run_icr.sh medqa artifacts/smoke_medqa
cat artifacts/smoke_medqa/analysis/summary.csv
```

Four items may contain no mixed-correctness pair; that is expected. Delete the
smoke root afterwards; it must not be reused for a full run.

---

## 6. Reproducing the main results (Qwen3-4B, Critical Evaluation)

One command per benchmark. Phase 1 generates 3 answers per item; phase 2
revises all six directions of every item on which not all three agents were
right, under the four main conditions.

```bash
export CUDA_DEVICES="0 1 2 3"      # the GPUs to use
for TASK in medqa arc_challenge gsm8k gpqa humanevalplus; do
  bash scripts/run_icr.sh $TASK artifacts/${TASK}_qwen3-4b_seed42
done
```

- Output: `artifacts/<task>_qwen3-4b_seed42/analysis/summary.csv` (per
  condition: records, CR, PR, SI, SR, SCR, 95% intervals, non-EOS and
  unparsed counts) and `paired_vs_no_message.csv`.
- Every step is resumable; rerun the same command after an interruption.
- Logs: `<root>/logs/`. `<root>/logs/verification.log` must end with
  `verification: PASS`.
- To speed up the non-latent conditions you may use `WORKERS_PER_GPU=2`, but
  run `true_latentmas` with `WORKERS_PER_GPU=1`, e.g. set
  `CONDITIONS=none,true_text,true_statebridge WORKERS_PER_GPU=2` first and then
  rerun with `CONDITIONS=true_latentmas WORKERS_PER_GPU=1`.
- Rough cost on 4 x RTX 5090: 2-4 GPU-hours per benchmark for phase 1 and a
  similar amount for phase 2 (GPQA-Diamond is the most expensive, about 6k
  tokens per generation).

### Reference values

Mixed-correctness strata of the reported runs (Qwen3-4B, `seed_pair_00`,
Critical Evaluation). Values are CR / PR / SI in percent; counts per stratum in
the last column.

| Benchmark | No Message | Answer Only | Full Text | StateBridge | LatentMAS | n (CR / PR) |
|---|---|---|---|---|---|---|
| MedQA | 6.90 / 98.28 / 52.59 | 38.79 / 80.17 / 59.48 | 80.17 / 42.24 / 61.21 | 54.31 / 71.55 / 62.93 | 69.83 / 32.76 / 51.29 | 116 / 116 |
| ARC-C | 8.16 / 92.86 / 50.51 | 38.78 / 63.27 / 51.02 | 80.61 / 34.69 / 57.65 | 69.39 / 44.90 / 57.14 | 58.16 / 45.92 / 52.04 | 98 / 98 |
| GSM8K | 13.04 / 92.39 / 52.72 | 39.13 / 70.65 / 54.89 | 51.09 / 52.17 / 51.63 | 39.13 / 73.91 / 56.52 | 59.78 / 46.74 / 53.26 | 92 / 92 |
| GPQA-D | 14.04 / 96.49 / 55.26 | 56.14 / 63.16 / 59.65 | 74.56 / 47.37 / 60.96 | 64.04 / 51.75 / 57.89 | 61.40 / 50.88 / 56.14 | 114 / 114 |
| HumanEval+ | 57.14 / 92.86 / 75.00 | - | 75.00 / 92.86 / 83.93 | 67.86 / 96.43 / 82.14 | 85.71 / 60.71 / 73.21 | 28 / 28 |

Re-running on the same GPU model with the same versions
reproduced records token for token in our checks; on other hardware expect
small differences from floating-point non-determinism.

---

## 7. Supplementary experiments

All supplementary runs reuse the phase 1 of a main run through
`scripts/derive_root.py`, so beliefs and StateBridge payloads are identical and
only the manipulated factor changes. The revision seed depends on
(item, direction, replication id) but not on the condition or the receiver
policy, so conditions and policies are compared on matched samples.

### 7.1 Answer Only (mixed-correctness directions)

```bash
TASK=medqa   # also arc_challenge, gsm8k, gpqa
python scripts/derive_root.py artifacts/${TASK}_qwen3-4b_seed42 artifacts/${TASK}_answer_only
CONDITIONS=true_answer CUDA_DEVICES="0 1" WORKERS_PER_GPU=2 \
  bash scripts/run_icr_mixed.sh $TASK artifacts/${TASK}_answer_only
PYTHONPATH=. python -m icr.analyze \
  --run ${TASK}=artifacts/${TASK}_qwen3-4b_seed42,artifacts/${TASK}_answer_only \
  --out artifacts/${TASK}_answer_only/analysis
```

### 7.2 Structured Verification receiver policy

```bash
# MedQA: all directions, all four channels
python scripts/derive_root.py artifacts/medqa_qwen3-4b_seed42 artifacts/medqa_verify
PYTHONPATH=. python -m icr.revisions --artifact-root artifacts/medqa_verify \
  --conditions none,true_text,true_statebridge,true_latentmas --receiver-policy verify \
  --skip-all-correct-items --both-correct-sample 1 --all-correct-sample 0
# GPQA-Diamond: mixed-correctness directions only
python scripts/derive_root.py artifacts/gpqa_qwen3-4b_seed42 artifacts/gpqa_verify
RECEIVER_POLICY=verify CUDA_DEVICES="0 1" \
  bash scripts/run_icr_mixed.sh gpqa artifacts/gpqa_verify
```

`icr.revisions` runs on the visible GPU as a single worker; to shard, start one
process per GPU with `--rank R --world-size W --global-resume` (this is what
the scripts do).

### 7.3 Revision-seed replication

```bash
python scripts/derive_root.py artifacts/medqa_qwen3-4b_seed42 artifacts/medqa_seed_pair_01 \
  --replication-id seed_pair_01
PYTHONPATH=. python -m icr.revisions --artifact-root artifacts/medqa_seed_pair_01 \
  --conditions true_statebridge --skip-all-correct-items --both-correct-sample 1 --all-correct-sample 0
```

### 7.4 Qwen3-8B (MedQA and GPQA-Diamond, mixed-correctness directions)

```bash
TASK=medqa   # or gpqa
MODEL=Qwen/Qwen3-8B PHASE1_ONLY=1 CUDA_DEVICES="0 1" \
  bash scripts/run_icr.sh $TASK artifacts/${TASK}_qwen3-8b_seed42
CONDITIONS=none,true_answer,true_text,true_statebridge,true_latentmas CUDA_DEVICES="0 1" \
  bash scripts/run_icr_mixed.sh $TASK artifacts/${TASK}_qwen3-8b_seed42
PYTHONPATH=. python -m icr.analyze --run 8B=artifacts/${TASK}_qwen3-8b_seed42 \
  --out artifacts/${TASK}_qwen3-8b_seed42/analysis
```

---

## 8. Protocol details that matter for exact reproduction

- **Generation.** Chat template with `enable_thinking=True`, a fixed system
  prompt, and `<think>` appended to the assistant turn. Sampling with
  temperature 0.6 and top-p 0.95; top-k is not passed, so the model's
  `generation_config.json` value (20) applies. Stop at `<|im_end|>` or
  `<|endoftext|>` or at `max_new_tokens` = 16,384 in both phases. bfloat16, the
  library-default attention implementation, batch size 1.
- **What is parsed and transmitted.** The thinking segment is stripped. The
  post-thinking response is parsed, shown to the receiver as its own prior
  reasoning, and sent verbatim by Full Text.
- **Seeds.** `transformers.set_seed` before every generation. Phase-1 seeds hash
  (global seed 42, replication id, item, agent); revision seeds hash
  (global seed, replication id, item, direction). Determinism is not enforced at
  the kernel level.
- **Phase-2 scope.** Items that all three agents answered correctly are skipped
  (`--skip-all-correct-items`); all six directions of every other item are
  revised (`--both-correct-sample 1`).
- **StateBridge payload.** Last-decoder-block states of the tokens after the first
  `</think>` (the whole generation if there is none), last 64 kept (fewer if the
  post-thinking segment is shorter), Procrustes-aligned to the embedding space,
  spliced at the message slot. LatentMAS instead prepends the sender's KV cache.
- **Scoring.** Multiple choice: last `\boxed{}` label, case-insensitive, must be
  one of a-d. GSM8K: first number in the last `\boxed{}` (else last number in
  the text), exact decimal comparison. HumanEval+: last ```` ```python ```` block
  executed with the dataset tests in a subprocess, 10 s wall-clock limit for the
  whole program. Non-terminating or unparsed outputs are scored wrong and kept.
- **Token budget of the reported runs.** The reported runs raised the budget in
  steps (2,048 -> 4,096 -> 8,192 -> 16,384, depending on the dataset) and
  regenerated only records that had not terminated. Sampling does not depend on
  the budget, so running at 16,384 from the start is expected to give the same
  output for every record that terminated under a smaller budget. Eight
  revision records of the reported runs (ARC-C 4, GSM8K 1, HumanEval+ 3) were
  left truncated at 2,048 or 4,096 tokens; a fresh run at 16,384 continues them.
- **HumanEval+ runs model-generated code** on the host without a sandbox. Run
  it in an isolated environment. Because the time limit is wall-clock, heavy
  CPU load can turn slow but correct programs into timeouts.

---

## 9. Run directory contents

```
<root>/config.json                  run configuration and fingerprint
<root>/prebeliefs/merged.jsonl      phase 1, one line per (item, agent)
<root>/messages/statebridge/...     StateBridge payloads (safetensors)
<root>/revisions/merged.jsonl       phase 2, one line per (item, direction, condition)
<root>/revisions/<shard>/records/   per-record JSON written by the workers
<root>/analysis/                    summary.csv, paired_vs_no_message.csv
```

Key record fields: `item_id`, `direction` (e.g. `A_to_B`, sender to receiver),
`condition`, `pair_classification`, `receiver_pre_answer`, `sender_pre_answer`,
`receiver_post_answer`, `receiver_post_correct`, `hit_eos`, `revision_seed`,
`prompt_sha256`.

---

## License

Apache License 2.0 (`LICENSE`). Third-party code and data attributions are in
`THIRD_PARTY_NOTICES.md`.
