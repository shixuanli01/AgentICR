"""Model-facing runtime that composes, but never modifies, StateBridge."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from safetensors.torch import save_file

from methods.state_bridge import StateBridge, strip_thinking
from models import ModelWrapper
from prompts import EMBEDDING_HINT_MARKER

from .channels import CommunicationMessage
from .prompts_v3 import external_block
from .protocol import (
    SYSTEM_PROMPT,
    independent_solver_prompt,
    parse_task_answer,
    revision_prompt,
    reset_rng,
    sha256_text,
)


def atomic_save_prefix(path: Path, prefix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
    save_file(
        {"statebridge_prefix": prefix.detach().cpu().contiguous()},
        str(temporary),
    )
    os.replace(temporary, path)


class ICRRuntime:
    def __init__(
        self,
        *,
        model_name: str,
        device: torch.device,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        task: str = "medqa",
    ) -> None:
        args = argparse.Namespace(
            model=model_name,
            model_name=model_name,
            task=task,
            prompt="sequential",
            batch_size=1,
        )
        self.model = ModelWrapper(model_name, device=device, args=args)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.task = task
        # V3 "revise" unless a run selects the V4 verify-then-decide policy.
        self.receiver_policy = "revise"
        self.bridge = StateBridge(
            self.model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_prefix_tokens=64,
            selection_method="last_k",
            enable_thinking=True,
            prefix_strategy="scale",
            adaptive_reg=1e-3,
            snap_ratio=0.3,
            use_hook=True,
            args=args,
        )
        self.embedding_layer = self.model.model.get_input_embeddings()
        self._embedding_mean_norm_value: Optional[torch.Tensor] = None

    def _embedding_mean_norm(self) -> torch.Tensor:
        """Mean row norm of the input embedding matrix, computed once.

        Upcasting the whole embedding matrix to float32 costs a multi-GiB
        transient allocation. The value is a constant of the frozen model, so
        it is computed on first use and reused for every LatentMAS handoff
        instead of being recomputed inside each revision.
        """
        if self._embedding_mean_norm_value is None:
            self._embedding_mean_norm_value = (
                self.embedding_layer.weight.detach().float().norm(dim=1).mean()
            )
            torch.cuda.empty_cache()
        return self._embedding_mean_norm_value

    def _render(self, user_content: str) -> str:
        prompt = self.model.render_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return f"{prompt}<think>"

    def _encode_prompt(
        self, prompt: str, prefix: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[int], str]:
        insert_position = None
        clean_prompt = prompt
        if prefix is not None:
            marker_index = prompt.find(EMBEDDING_HINT_MARKER)
            if marker_index < 0:
                raise RuntimeError("Latent revision prompt is missing its marker")
            left_text = prompt[:marker_index]
            insert_position = len(
                self.model.tokenizer(left_text, add_special_tokens=False)["input_ids"]
            )
            clean_prompt = prompt.replace(EMBEDDING_HINT_MARKER, "")
        elif EMBEDDING_HINT_MARKER in prompt:
            raise RuntimeError("A non-latent prompt contains the embedding marker")

        encoded = self.model.tokenizer(
            clean_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return input_ids, attention_mask, insert_position, clean_prompt

    def _post_think_slice(
        self, hidden: torch.Tensor, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end_ids = self.model.tokenizer.encode("</think>", add_special_tokens=False)
        end_position = -1
        tokens = token_ids[0].tolist()
        if end_ids:
            width = len(end_ids)
            for index in range(len(tokens) - width + 1):
                if tokens[index : index + width] == end_ids:
                    end_position = index + width
                    break
        if 0 < end_position < hidden.shape[1]:
            return hidden[:, end_position:, :], token_ids[:, end_position:]
        return hidden, token_ids

    @torch.no_grad()
    def generate_prebelief(
        self, question: str, *, seed: int
    ) -> dict[str, Any]:
        reset_rng(seed)
        user_content = independent_solver_prompt(self.task, question)
        prompt = self._render(user_content)
        input_ids, attention_mask, _, clean_prompt = self._encode_prompt(prompt, None)
        prompt_embeds = self.embedding_layer(input_ids)
        started = time.time()
        texts, hidden, token_ids, generation_info = self.bridge._generate_with_prefix(
            prompt_embeds,
            attention_mask,
            prefix_embeds=None,
            insert_position=None,
            need_hidden_states=True,
        )
        torch.cuda.synchronize(self.device)
        generation_seconds = time.time() - started
        raw_response = self.model.tokenizer.decode(
            token_ids[0], skip_special_tokens=True
        ).strip()
        reasoning_text = texts[0]
        filtered_hidden, filtered_ids = self._post_think_slice(hidden, token_ids)
        if filtered_hidden.shape[1] == 0:
            raise RuntimeError("Independent solve produced no states for StateBridge")
        alignment_started = time.time()
        prefix, diagnostic = self.bridge._prepare_handoff(
            raw_hidden=hidden,
            raw_token_ids=token_ids,
            filtered_hidden=filtered_hidden,
            filtered_token_ids=filtered_ids,
            incoming_prefix=None,
            agent_role="planner",
        )
        torch.cuda.synchronize(self.device)
        alignment_seconds = time.time() - alignment_started
        result = {
            "reasoning_text": reasoning_text,
            "raw_response": raw_response,
            "parsed_answer": parse_task_answer(self.task, reasoning_text),
            "generation_seed": seed,
            "generated_token_ids": token_ids[0].detach().cpu().tolist(),
            "generation_length": int(token_ids.shape[1]),
            "hit_eos": bool(generation_info["hit_eos"]),
            "prompt_tokens": int(attention_mask.sum()),
            "prompt_sha256": sha256_text(clean_prompt),
            "generation_seconds": generation_seconds,
            "alignment_seconds": alignment_seconds,
            "statebridge": diagnostic,
        }
        del hidden, filtered_hidden, filtered_ids, prompt_embeds
        torch.cuda.empty_cache()
        return {"record": result, "prefix": prefix.detach().cpu()}

    def build_revision_prompt(
        self,
        *,
        question: str,
        receiver_reasoning: str,
        receiver_answer: Optional[str],
        message: CommunicationMessage,
    ) -> str:
        # ICR-V3: every condition shares one template and one message slot, which
        # sits between the receiver's prior and the integration rules. Only the
        # payload differs, so the visible text of the StateBridge and LatentMAS
        # conditions is byte-identical and the Text condition differs from them
        # only in the message body.
        #
        # The StateBridge prefix is spliced at the marker's token position. The
        # LatentMAS KV cache cannot be: its RoPE positions are baked in on the
        # sender side and `generate` has no mid-sequence cache API, so it stays a
        # front-anchored causal prefix. That asymmetry is a declared comparison
        # boundary of the protocol.
        if message.condition == "none":
            block = external_block("none")
        elif message.text is not None:
            block = external_block(message.condition, sender_reasoning=message.text)
        elif message.prefix is not None or message.trajectory is not None:
            block = external_block(message.condition)
        else:
            raise RuntimeError("Non-none communication message has no payload")
        builder = revision_prompt
        if getattr(self, "receiver_policy", "revise") == "verify":
            from .prompts_v4 import revision_prompt_v4 as builder
        user_content = builder(
            getattr(self, "task", "medqa"),
            question=question,
            receiver_prior_reasoning=receiver_reasoning,
            receiver_prior_answer=(receiver_answer or "UNPARSEABLE"),
            external_block=block,
        )
        return self._render(user_content)

    def _latentmas_source_ids(
        self, source: Mapping[str, Any]
    ) -> tuple[torch.Tensor, int, int, str]:
        """Rebuild the exact cached Phase-1 token trajectory without sampling."""
        sender_prompt = self._render(
            independent_solver_prompt(self.task, str(source["question"]))
        )
        encoded_prompt = self.model.tokenizer(
            sender_prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        if int(encoded_prompt.shape[1]) != int(source["prompt_tokens"]):
            raise RuntimeError("Cached sender prompt token count no longer matches")
        if sha256_text(sender_prompt) != source["prompt_sha256"]:
            raise RuntimeError("Cached sender prompt hash no longer matches")
        generated = torch.tensor(
            [source["generated_token_ids"]], dtype=torch.long
        )
        ids = torch.cat((encoded_prompt, generated), dim=1).to(self.device)
        return ids, int(encoded_prompt.shape[1]), int(generated.shape[1]), sender_prompt

    @staticmethod
    def _past_length(past_key_values: Any) -> int:
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        return int(past_key_values[0][0].shape[-2])

    @staticmethod
    def _kv_payload(past_key_values: Any) -> tuple[int, int, str]:
        legacy = (
            past_key_values.to_legacy_cache()
            if hasattr(past_key_values, "to_legacy_cache")
            else past_key_values
        )
        tensors = [tensor for layer in legacy for tensor in layer[:2]]
        payload_bytes = sum(t.numel() * t.element_size() for t in tensors)
        return int(payload_bytes), len(legacy), str(tensors[0].dtype).replace("torch.", "")

    @torch.no_grad()
    def generate_latentmas_revision(
        self,
        *,
        question: str,
        receiver_reasoning: str,
        receiver_answer: Optional[str],
        message: CommunicationMessage,
        seed: int,
        latent_steps: int = 10,
    ) -> dict[str, Any]:
        """Official LatentMAS KV relay adapted to one frozen ICR trajectory."""
        if message.trajectory is None:
            raise RuntimeError("LatentMAS requires a cached source trajectory")
        reset_rng(seed)
        source_ids, source_prompt_tokens, source_generated_tokens, _ = (
            self._latentmas_source_ids(message.trajectory)
        )
        source_mask = torch.ones_like(source_ids, device=self.device)
        torch.cuda.synchronize(self.device)
        communication_started = time.time()
        outputs = self.model.model(
            input_ids=source_ids,
            attention_mask=source_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            # Only the last position's hidden state is used, so materializing
            # vocabulary logits for every source token wastes several GiB on a
            # long sender trajectory. The cache and hidden states are unchanged.
            logits_to_keep=1,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        # The official implementation applies identity realignment followed by
        # input-embedding mean-norm matching when realignment is disabled.
        target_norm = self._embedding_mean_norm()
        for _ in range(latent_steps):
            latent = last_hidden.float()
            latent = latent * (
                target_norm / latent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            )
            latent = latent.to(self.embedding_layer.weight.dtype).unsqueeze(1)
            past_len = self._past_length(past)
            latent_mask = torch.ones(
                (1, past_len + 1), dtype=torch.long, device=self.device
            )
            outputs = self.model.model(
                inputs_embeds=latent,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
        torch.cuda.synchronize(self.device)
        communication_seconds = time.time() - communication_started
        payload_bytes, cache_layers, cache_dtype = self._kv_payload(past)
        cache_tokens = self._past_length(past)

        prompt = self.build_revision_prompt(
            question=question,
            receiver_reasoning=receiver_reasoning,
            receiver_answer=receiver_answer,
            message=message,
        )
        clean_prompt = prompt.replace(EMBEDDING_HINT_MARKER, "")
        encoded = self.model.tokenizer(
            clean_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(self.device)
        prompt_mask = encoded["attention_mask"].to(self.device)
        prompt_tokens = int(prompt_mask.sum())
        attention_mask = torch.cat(
            (
                torch.ones((1, cache_tokens), dtype=prompt_mask.dtype, device=self.device),
                prompt_mask,
            ),
            dim=-1,
        )
        # transformers 4.51 derives cache_position from input_ids and then
        # discards the already-cached prefix. Supply inert placeholder IDs for
        # those positions; they are never forwarded through the model.
        cached_placeholders = torch.full(
            (1, cache_tokens),
            int(self.model.tokenizer.pad_token_id),
            dtype=input_ids.dtype,
            device=self.device,
        )
        generation_input_ids = torch.cat((cached_placeholders, input_ids), dim=1)
        reset_rng(seed)
        torch.cuda.synchronize(self.device)
        generation_started = time.time()
        generated = self.model.model.generate(
            input_ids=generation_input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.model.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        torch.cuda.synchronize(self.device)
        generation_seconds = time.time() - generation_started
        generated_ids = generated.sequences[:, generation_input_ids.shape[1]:]
        raw_response = self.model.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()
        response = strip_thinking(raw_response)
        last_token = int(generated_ids[0, -1]) if generated_ids.shape[1] else None
        result = {
            "response": response,
            "raw_response": raw_response,
            "parsed_answer": parse_task_answer(self.task, response),
            "revision_seed": seed,
            "generated_token_ids": generated_ids[0].detach().cpu().tolist(),
            "generation_length": int(generated_ids.shape[1]),
            "hit_eos": last_token == self.model.tokenizer.eos_token_id,
            "prompt_tokens": prompt_tokens,
            "prompt_sha256": sha256_text(clean_prompt),
            "generation_seconds": generation_seconds,
            "latentmas_compute": {
                "teacher_forcing_forward_steps": 1,
                "generated_latent_steps": latent_steps,
                "total_communication_forward_steps": 1 + latent_steps,
                "source_prompt_tokens": source_prompt_tokens,
                "source_generated_tokens": source_generated_tokens,
                "cache_positions": cache_tokens,
                "cache_layers": cache_layers,
                "cache_dtype": cache_dtype,
                "kv_payload_bytes": payload_bytes,
                "communication_seconds": communication_seconds,
                "receiver_revision_generations": 1,
            },
        }
        del source_ids, source_mask, outputs, past, input_ids, prompt_mask
        del cached_placeholders, generation_input_ids
        del attention_mask, generated, last_hidden
        torch.cuda.empty_cache()
        return result

    @torch.no_grad()
    def generate_revision(
        self,
        *,
        question: str,
        receiver_reasoning: str,
        receiver_answer: Optional[str],
        message: CommunicationMessage,
        seed: int,
    ) -> dict[str, Any]:
        reset_rng(seed)
        prompt = self.build_revision_prompt(
            question=question,
            receiver_reasoning=receiver_reasoning,
            receiver_answer=receiver_answer,
            message=message,
        )
        prefix = message.prefix.to(self.device) if message.prefix is not None else None
        input_ids, attention_mask, insert_position, clean_prompt = self._encode_prompt(
            prompt, prefix
        )
        prompt_embeds = self.embedding_layer(input_ids)
        if prefix is not None:
            assert insert_position is not None
            prefix_mask = torch.ones(
                (attention_mask.shape[0], prefix.shape[1]),
                dtype=attention_mask.dtype,
                device=self.device,
            )
            prompt_embeds = torch.cat(
                (
                    prompt_embeds[:, :insert_position],
                    prefix,
                    prompt_embeds[:, insert_position:],
                ),
                dim=1,
            )
            attention_mask = torch.cat(
                (
                    attention_mask[:, :insert_position],
                    prefix_mask,
                    attention_mask[:, insert_position:],
                ),
                dim=1,
            )

        started = time.time()
        generated = self.model.model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.model.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        torch.cuda.synchronize(self.device)
        seconds = time.time() - started
        generated_ids = generated.sequences
        raw_response = self.model.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()
        response = strip_thinking(raw_response)
        last_token = int(generated_ids[0, -1]) if generated_ids.shape[1] else None
        result = {
            "response": response,
            "raw_response": raw_response,
            "parsed_answer": parse_task_answer(self.task, response),
            "revision_seed": seed,
            "generated_token_ids": generated_ids[0].detach().cpu().tolist(),
            "generation_length": int(generated_ids.shape[1]),
            "hit_eos": last_token == self.model.tokenizer.eos_token_id,
            "prompt_tokens": int(attention_mask.sum()),
            "prompt_sha256": sha256_text(clean_prompt),
            "generation_seconds": seconds,
        }
        del prompt_embeds, input_ids, attention_mask, generated
        if prefix is not None:
            del prefix
        torch.cuda.empty_cache()
        return result
