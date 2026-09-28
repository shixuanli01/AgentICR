# Third-Party Notices

## StateBridge

The StateBridge channel (`methods/state_bridge.py`) and the shared model,
data-loading, prompt and utility code (`models.py`, `data.py`, `prompts.py`,
`utils.py`, `methods/__init__.py`) are taken from the public StateBridge
release and modified for this evaluation.

- Upstream: https://github.com/YanwenPneg/StateBridge
- Commit: `3f6bf5442c6e8848555a6132516e6d36f35444fb`
- License: Apache License 2.0

## LatentMAS

Portions of the StateBridge code above are in turn adapted from LatentMAS
(https://github.com/Gen-Verse/LatentMAS), Apache License 2.0. The LatentMAS
communication channel in `icr/runtime.py` follows the official LatentMAS KV
relay (upstream commit `9a9e4d331eb11430bd9e64754c6b252b06d73031`).

## Datasets and model

No dataset or model weights are distributed in this repository.

- MedQA (https://github.com/jind11/MedQA), MIT License. The 300-question
  subset is obtained from the StateBridge release (see `data/README.md`).
- GPQA (https://github.com/idavidrein/gpqa), CC BY 4.0. The transformed
  GPQA-Diamond file is obtained from the StateBridge release.
- ARC-Challenge (`allenai/ai2_arc`), GSM8K (`openai/gsm8k`) and HumanEval+
  (`evalplus/humanevalplus`) are downloaded from Hugging Face at run time and
  remain under their own terms.
- Qwen3 model weights are downloaded from Hugging Face and remain governed by
  their model card.
