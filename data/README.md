# Data

Two evaluation files must be placed in this directory before running MedQA or
GPQA-Diamond. They are the exact files used for the reported results and are
distributed with the public StateBridge release; they are not stored here.

```bash
# from the repository root
COMMIT=3f6bf5442c6e8848555a6132516e6d36f35444fb
curl -L -o data/medqa.json        https://raw.githubusercontent.com/YanwenPneg/StateBridge/$COMMIT/data/medqa.json
curl -L -o data/gpqa_diamond.json https://raw.githubusercontent.com/YanwenPneg/StateBridge/$COMMIT/data/gpqa_diamond.json
sha256sum data/medqa.json data/gpqa_diamond.json
```

Expected SHA-256:

| File | SHA-256 |
|---|---|
| `medqa.json` | `d4f09708b623a7750013b22c607495eb8872bbef7a3f50990b7279c8fd87ec8d` |
| `gpqa_diamond.json` | `17c1db2ce55e9c18de41aa8e8eb91f41f0eaef19815b9c1b900ed8e83df7a8c5` |

- `medqa.json`: 300 USMLE questions from MedQA with fields `query`, `options`,
  `answer`. Item IDs used throughout are the 0-based row indices (0-299).
- `gpqa_diamond.json`: all 198 GPQA-Diamond questions, choices relabelled A-D,
  fields `question`, `answer`. Item IDs are the 0-based row indices.

ARC-Challenge, GSM8K and HumanEval+ need no manual download; they are loaded
from Hugging Face (`allenai/ai2_arc` config `ARC-Challenge` split `test`,
`gsm8k` config `main` split `test`, `evalplus/humanevalplus` split `test`).

Every run writes `dataset_sha256` (a hash of the loaded benchmark rows) into
its `config.json`. The values of the reported runs are:

| Task | `dataset_sha256` |
|---|---|
| medqa | `ddee8dc64d3b2a1e5061c409c68d626109237640009168838a038db29efe802c` |
| arc_challenge | `e5f26d489a59a1cf0e77ed86ce732a321026a13d294d25582bfcff2129372be6` |
| gsm8k | `14dbb509584329a3fe0aff683f095cc9a951e1dd8d7fbe024a60351ee4be5a45` |
| gpqa | `42cdb10747fbb798116e3b490f470836d6ec1d47d24ae5b06dea7e8cae616f42` |
| humanevalplus | `2669a4b46e251fe9198fcf811dbdbeff6e7d184622ca4891b57a72cd5342dfd9` |

A different value means the downloaded data differ from the reported runs.
