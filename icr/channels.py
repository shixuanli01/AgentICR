"""Extensible communication-channel interface for ICR."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from typing import Any, Mapping, Optional

from safetensors.torch import load_file


@dataclass
class CommunicationMessage:
    condition: str
    source_item_id: Optional[int]
    source_agent_id: Optional[str]
    text: Optional[str] = None
    prefix: Optional[torch.Tensor] = None
    trajectory: Optional[Mapping[str, Any]] = None
    diagnostics: Optional[dict[str, Any]] = None


class CommunicationChannel(ABC):
    """A channel constructs a payload without changing revision semantics."""

    condition: str

    @abstractmethod
    def build_message(
        self,
        sender_record: Mapping[str, Any],
        receiver_record: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> CommunicationMessage:
        raise NotImplementedError

    def apply_to_receiver(
        self, receiver_inputs: Mapping[str, Any], message: CommunicationMessage
    ) -> dict[str, Any]:
        return {**receiver_inputs, "communication_message": message}


class NoCommunicationChannel(CommunicationChannel):
    condition = "none"

    def build_message(self, sender_record, receiver_record, context):
        del sender_record, receiver_record, context
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=None,
            source_agent_id=None,
            diagnostics={"modality": "none", "payload_bytes": 0},
        )


class TextCommunicationChannel(CommunicationChannel):
    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        if self.source == "true":
            source = sender_record
        elif self.source == "self":
            source = receiver_record
        elif self.source == "other":
            source = context["other_sender_record"]
        else:
            raise ValueError(f"Unknown text source: {self.source}")
        text = str(source["reasoning_text"])
        tokenizer = context["tokenizer"]
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            text=text,
            diagnostics={
                "modality": "text",
                "tokens": token_count,
                "characters": len(text),
                "payload_bytes": len(text.encode("utf-8")),
            },
        )


class AnswerOnlyCommunicationChannel(CommunicationChannel):
    """The sender's final answer, normalised, and nothing else.

    It separates two things Full Text conflates: knowing what another process
    concluded, and seeing why. Whatever Full Text achieves over this condition
    is what the reasoning itself buys.

    A sender whose answer did not parse sends a fixed placeholder and stays in
    the population; dropping those records only for this condition would change
    its denominator relative to every other channel.
    """

    PLACEHOLDER = "UNPARSEABLE"

    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        source = {"true": sender_record, "self": receiver_record}.get(self.source)
        if source is None:
            source = context["other_sender_record"]
        answer = source.get("parsed_answer")
        unparsed = answer is None
        text = self.PLACEHOLDER if unparsed else str(answer).strip().upper()
        tokenizer = context["tokenizer"]
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            text=text,
            diagnostics={
                "modality": "answer_only",
                "sender_answer_unparsed": unparsed,
                "tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                "characters": len(text),
                "payload_bytes": len(text.encode("utf-8")),
                "payload_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
        )


class StateBridgeCommunicationChannel(CommunicationChannel):
    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        if self.source == "true":
            source = sender_record
        elif self.source == "self":
            source = receiver_record
        elif self.source == "other":
            source = context["other_sender_record"]
        else:
            raise ValueError(f"Unknown StateBridge source: {self.source}")
        prefix_path = Path(context["artifact_root"]) / source["statebridge_prefix_file"]
        prefix = load_file(str(prefix_path), device="cpu")["statebridge_prefix"]
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            prefix=prefix,
            diagnostics={
                "modality": "statebridge",
                "states": int(prefix.shape[1]),
                "hidden_dimension": int(prefix.shape[2]),
                "dtype": str(prefix.dtype).replace("torch.", ""),
                "payload_bytes": int(prefix.numel() * prefix.element_size()),
            },
        )


class LatentMASCommunicationChannel(CommunicationChannel):
    """Reference an exact cached Phase-1 trajectory for KV reconstruction."""

    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        if self.source == "true":
            source = sender_record
        elif self.source == "self":
            source = receiver_record
        elif self.source == "other":
            source = context["other_sender_record"]
        else:
            raise ValueError(f"Unknown LatentMAS source: {self.source}")
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            trajectory=source,
            diagnostics={
                "modality": "latentmas",
                "source_prompt_sha256": source["prompt_sha256"],
                "source_generation_seed": int(source["generation_seed"]),
                "source_generated_tokens": len(source["generated_token_ids"]),
                "cache_storage": "reconstructible_reference",
            },
        )


def make_channel(condition: str) -> CommunicationChannel:
    if condition == "none":
        return NoCommunicationChannel()
    if condition.endswith("_text"):
        return TextCommunicationChannel(condition, condition.removesuffix("_text"))
    if condition.endswith("_answer"):
        return AnswerOnlyCommunicationChannel(condition, condition.removesuffix("_answer"))
    if condition.endswith("_statebridge"):
        return StateBridgeCommunicationChannel(
            condition, condition.removesuffix("_statebridge")
        )
    if condition.endswith("_latentmas"):
        return LatentMASCommunicationChannel(
            condition, condition.removesuffix("_latentmas")
        )
    raise ValueError(f"Unknown condition: {condition}")
