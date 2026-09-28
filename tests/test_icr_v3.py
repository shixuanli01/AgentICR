"""ICR-V3 prompt alignment, per-task output contracts, and scoring fixes."""

import pytest

from icr.parsing_v3 import (
    extract_numeric_gold,
    numeric_equal,
    parse_choice_answer,
    parse_numeric_answer,
)
from icr.prompts_v3 import (
    ANSWER_FORMAT,
    PROMPT_VERSION,
    external_block,
    revision_prompt,
)
from prompts import EMBEDDING_HINT_MARKER


PRIOR = {
    "question": "<<QUESTION>>",
    "receiver_prior_reasoning": "<<PRIOR>>",
    "receiver_prior_answer": "a",
}


def _revision(task, condition, sender_reasoning=None):
    return revision_prompt(
        task,
        external_block_text=external_block(
            condition, sender_reasoning=sender_reasoning
        ),
        **PRIOR,
    )


# --- Scoring fixes S1-S5 -------------------------------------------------


def test_s1_numeric_gold_keeps_thousands_separator_digits():
    # Legacy utils.extract_gold truncated "#### 2,125" to "2".
    assert extract_numeric_gold("Work.\n#### 2,125") == "2125"
    assert extract_numeric_gold("#### 18") == "18"
    assert extract_numeric_gold("no gold here") is None


def test_s2_boxed_number_survives_thousands_separator():
    # Legacy parsing returned "2" for the comma form, inverting correctness.
    assert parse_numeric_answer(r"So the answer is \boxed{2,125}.") == "2125"
    assert parse_numeric_answer(r"So the answer is \boxed{2125}.") == "2125"
    assert numeric_equal(
        parse_numeric_answer(r"\boxed{2125}"), extract_numeric_gold("#### 2,125")
    )
    assert numeric_equal(
        parse_numeric_answer(r"\boxed{2,125}"), extract_numeric_gold("#### 2,125")
    )


def test_s3_boxed_label_with_trailing_option_text():
    assert parse_choice_answer(r"\boxed{C. Colorectal cancer}") == "c"
    assert parse_choice_answer(r"\boxed{(B)}") == "b"
    assert parse_choice_answer(r"\boxed{d}") == "d"


def test_s4_boxed_label_inside_latex_wrapper():
    assert parse_choice_answer(r"\boxed{\text{C}}") == "c"
    assert parse_numeric_answer(r"\boxed{\text{42}}") == "42"


def test_s5_numeric_equality_ignores_pure_formatting():
    assert numeric_equal("18.0", "18")
    assert numeric_equal("+18", "18")
    assert not numeric_equal("18", "19")
    assert not numeric_equal(None, "18")


def test_choice_parser_rejects_out_of_contract_answers():
    assert parse_choice_answer(r"\boxed{E}") is None
    assert parse_choice_answer("The answer is clearly C.") is None
    assert parse_choice_answer(r"\boxed{}") is None


def test_numeric_parser_falls_back_to_last_number_without_boxed():
    assert parse_numeric_answer("first 3 then 7") == "7"


# --- Condition alignment -------------------------------------------------


def test_statebridge_and_latentmas_prompts_are_byte_identical():
    assert _revision("medqa", "true_statebridge") == _revision(
        "medqa", "true_latentmas"
    )


def test_text_differs_from_latent_only_in_the_message_body():
    latent = _revision("medqa", "true_statebridge")
    text = _revision("medqa", "true_text", sender_reasoning="<<SENDER>>")
    assert text == latent.replace(EMBEDDING_HINT_MARKER, "<<SENDER>>")


def test_none_differs_from_latent_only_in_the_external_block():
    latent = _revision("medqa", "true_statebridge")
    none = _revision("medqa", "none")
    assert none == latent.replace(
        external_block("true_statebridge"), external_block("none")
    )


def test_message_slot_sits_between_prior_and_integration_rules():
    prompt = _revision("medqa", "true_statebridge")
    assert (
        prompt.index("Your previous answer:")
        < prompt.index(EMBEDDING_HINT_MARKER)
        < prompt.index("Your task is to REVISE")
    )


def test_self_and_other_controls_share_the_true_condition_rendering():
    for suffix in ("statebridge", "latentmas"):
        assert external_block(f"self_{suffix}") == external_block(f"true_{suffix}")
        assert external_block(f"other_{suffix}") == external_block(f"true_{suffix}")


# --- Per-task output contracts -------------------------------------------


@pytest.mark.parametrize(
    "task", ["medqa", "gpqa", "arc_challenge", "gsm8k", "humanevalplus"]
)
def test_the_revision_prompt_closes_with_its_task_output_contract(task):
    # Phase 1 carries V2's own contract; only phase 2 uses these.
    assert _revision(task, "none").endswith(ANSWER_FORMAT[task])


def test_answer_contracts_match_the_labels_each_dataset_displays():
    assert "one of A, B, C, or D" in ANSWER_FORMAT["medqa"]
    assert "a, b, c, or d" in ANSWER_FORMAT["arc_challenge"]
    assert "final labeled list" in ANSWER_FORMAT["gpqa"]
    assert "no thousands separators" in ANSWER_FORMAT["gsm8k"]
    assert "markdown Python code block" in ANSWER_FORMAT["humanevalplus"]


def test_code_tasks_use_the_implementation_revision_template():
    prompt = _revision("humanevalplus", "true_text", sender_reasoning="<<SENDER>>")
    assert "Your previous reasoning and implementation:" in prompt
    assert "REVISE your implementation" in prompt


def test_prompt_version_is_recorded():
    assert PROMPT_VERSION == "icr_v3_mid_injection"


# --- Structural exclusion ------------------------------------------------


def test_arc_structural_exclusion_counts_trailing_options():
    from icr.benchmarks import structural_exclusions, trailing_option_labels

    four = "Stem?\na: one\nb: two\nc: three\nd: four"
    five = "Stem?\na: one\nb: two\nc: three\nd: four\ne: five"
    three = "Stem?\na: one\nb: two\nc: three"
    assert trailing_option_labels(four) == ["a", "b", "c", "d"]
    assert trailing_option_labels(five) == ["a", "b", "c", "d", "e"]

    data = [{"question": four}, {"question": five}, {"question": three}]
    assert structural_exclusions("arc_challenge", data) == [1, 2]
    assert structural_exclusions("gsm8k", data) == []


# --- Phase 1 is V2 verbatim -----------------------------------------------


def test_phase1_prompts_are_byte_identical_to_v2():
    """V3 rewrote phase 1 and asked for concise reasoning, which V2 never did.

    Independent-solve accuracy fell and the A/B disagreement subset shrank, so
    phase 1 is restored verbatim. These assertions keep it that way.
    """
    from icr import protocol

    assert protocol.independent_solver_prompt(
        "medqa", "Q?"
    ) == protocol.INDEPENDENT_SOLVER_PROMPT.format(question="Q?")
    assert protocol.independent_solver_prompt(
        "gsm8k", "Q?"
    ) == protocol.NUMERIC_INDEPENDENT_SOLVER_PROMPT.format(question="Q?")
    assert protocol.independent_solver_prompt(
        "arc_challenge", "Q?"
    ) == protocol.GENERAL_INDEPENDENT_SOLVER_PROMPT.format(question="Q?")
    assert protocol.independent_solver_prompt(
        "humanevalplus", "Q?"
    ) == protocol.CODE_INDEPENDENT_SOLVER_PROMPT.format(question="Q?")


def test_phase1_never_asks_for_concise_reasoning():
    from icr import protocol

    for task in ("medqa", "gpqa", "arc_challenge", "gsm8k", "humanevalplus"):
        assert "concise" not in protocol.independent_solver_prompt(task, "Q?").lower()


def test_gpqa_is_the_only_phase1_deviation_and_disambiguates_its_layers():
    from icr import protocol

    gpqa = protocol.independent_solver_prompt("gpqa", "Q?")
    general = protocol.GENERAL_INDEPENDENT_SOLVER_PROMPT.format(question="Q?")
    assert gpqa != general
    assert gpqa == general.replace(
        "replacing A with one of A, B, C, or D.",
        "replacing A with one of A, B, C, or D." + protocol.GPQA_LAYER_DISAMBIGUATION,
        1,
    )
    assert "final A-D list" in gpqa


def test_jsonl_readers_survive_unicode_line_separators(tmp_path):
    """A record may contain U+2028, which str.splitlines() treats as a break.

    json.dumps leaves that character unescaped, so a merged.jsonl written from
    GPQA prebeliefs splits into more "lines" than it has records and the
    fragments do not parse. Every jsonl reader must split on newlines only.
    """
    import json

    record = {"item_id": 0, "agent_id": "A", "reasoning_text": "before after"}
    path = tmp_path / "merged.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    text = path.read_text(encoding="utf-8")
    assert len([l for l in text.splitlines() if l]) == 2  # the trap
    rows = [json.loads(l) for l in text.split("\n") if l]
    assert len(rows) == 1
    assert rows[0]["reasoning_text"] == "before after"


# --- Selective rerun of truncated records --------------------------------


def test_repair_shards_over_truncated_records_and_keeps_their_paths(tmp_path):
    """A repair must balance across workers and not leave duplicate records.

    Striding over every item leaves most workers idle when only a handful of
    records need regenerating. Sharding over the truncated records fixes that,
    but each one has to be written back to the rank directory it already lives
    in, or the same (item, agent) would exist under two ranks.
    """
    import json

    root = tmp_path / "prebeliefs"
    for rank, (item, agent, eos) in enumerate(
        [(3, "A", False), (7, "B", False), (5, "A", True), (9, "B", False)]
    ):
        d = root / f"rank{rank}" / "records"
        d.mkdir(parents=True)
        (d / f"item_{item:04d}_{agent}.json").write_text(
            json.dumps({"item_id": item, "agent_id": agent, "hit_eos": eos}),
            encoding="utf-8",
        )

    truncated = {}
    for path in root.glob("rank*/records/item_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not row.get("hit_eos", True):
            truncated[(int(row["item_id"]), str(row["agent_id"]))] = path

    assert set(truncated) == {(3, "A"), (7, "B"), (9, "B")}
    assert (5, "A") not in truncated  # reached EOS, must not be regenerated

    items = sorted({item for item, _ in truncated})
    world = 2
    assert sorted(items[0::world] + items[1::world]) == items  # every item covered
    assert not set(items[0::world]) & set(items[1::world])  # and covered once

    # Each record keeps the rank directory it was found in.
    assert truncated[(3, "A")].parts[-3] == "rank0"
    assert truncated[(7, "B")].parts[-3] == "rank1"
    assert truncated[(9, "B")].parts[-3] == "rank3"


# --- Third agent -----------------------------------------------------------


def test_three_agents_give_six_ordered_pairs():
    from icr import AGENTS, DIRECTIONS, direction_agents

    assert AGENTS == ("A", "B", "C")
    assert len(DIRECTIONS) == 6
    assert set(DIRECTIONS) == {
        "A_to_B", "A_to_C", "B_to_A", "B_to_C", "C_to_A", "C_to_B"
    }
    for direction in DIRECTIONS:
        sender, receiver = direction_agents(direction)
        assert sender != receiver
        assert {sender, receiver} <= set(AGENTS)


def test_direction_agents_rejects_nonsense():
    import pytest

    from icr import direction_agents

    for bad in ("A_to_A", "A_to_D", "D_to_A", "AtoB", ""):
        with pytest.raises(ValueError):
            direction_agents(bad)


def test_adding_agent_c_leaves_the_existing_seeds_untouched():
    """The A and B beliefs already generated must stay valid.

    prebelief_seed hashes the agent id, and revision_seed hashes the direction
    label, so a third agent adds new seeds without disturbing the old ones. If
    this broke, every belief generated so far would have to be discarded.
    """
    from icr.protocol import prebelief_seed, revision_seed

    assert prebelief_seed(42, 0, "A", "seed_pair_00") == 1352253858
    assert prebelief_seed(42, 0, "B", "seed_pair_00") == 1804839786
    assert revision_seed(42, 0, "A_to_B", "seed_pair_00") == 1771418669
    # The new agent and directions are simply different draws.
    assert prebelief_seed(42, 0, "C", "seed_pair_00") not in {
        prebelief_seed(42, 0, "A", "seed_pair_00"),
        prebelief_seed(42, 0, "B", "seed_pair_00"),
    }


def test_a_mixed_item_yields_twice_the_correction_and_destruction_cases():
    """Why a third agent is worth more than a third replication.

    With two agents a mixed item gives one correction case and one destruction
    case. With three it gives two of each, whichever way the correctness falls,
    and items where the first two agreed can now turn out mixed.
    """
    from icr import DIRECTIONS, direction_agents
    from icr.protocol import classify_pair

    for correct in ({"A": True, "B": False, "C": False},
                    {"A": True, "B": True, "C": False}):
        counts = {"correction_opportunity": 0, "destruction_risk": 0}
        for direction in DIRECTIONS:
            sender, receiver = direction_agents(direction)
            category = classify_pair(correct[sender], correct[receiver])
            if category in counts:
                counts[category] += 1
        assert counts == {"correction_opportunity": 2, "destruction_risk": 2}


def test_skipping_all_correct_items_still_leaves_scr_measurable():
    """Items every agent got right cannot show anything, so they are skipped.

    A mixed item still contains two both-correct pairs, so preservation on
    already-correct answers stays measured without paying for the items where
    communication has no room to act. LatentMAS destroyed 4 of 378 such pairs on
    MedQA, so this is not a quantity that can simply be assumed to be 100%.
    """
    from icr import DIRECTIONS, direction_agents
    from icr.protocol import classify_pair

    def categories(correct):
        counts = {}
        for direction in DIRECTIONS:
            sender, receiver = direction_agents(direction)
            category = classify_pair(correct[sender], correct[receiver])
            counts[category] = counts.get(category, 0) + 1
        return counts

    all_correct = categories({"A": True, "B": True, "C": True})
    assert all_correct == {"both_correct": 6}  # nothing to learn, skip the item

    two_correct = categories({"A": True, "B": True, "C": False})
    assert two_correct == {
        "both_correct": 2,       # SCR survives inside a mixed item
        "correction_opportunity": 2,
        "destruction_risk": 2,
    }

    one_correct = categories({"A": True, "B": False, "C": False})
    assert one_correct == {
        "both_wrong": 2,         # and SR does too
        "correction_opportunity": 2,
        "destruction_risk": 2,
    }

    assert categories({"A": False, "B": False, "C": False}) == {"both_wrong": 6}


def test_resume_shards_over_outstanding_pairs_not_every_pair():
    """A dead worker's backlog must not land on one worker again.

    Shards are strides, and a stride nests inside any divisor of its world
    size: pairs[7::8] sits entirely inside pairs[3::4] and pairs[1::2]. Re-running
    a failed world-8 run at world 4 or 2 therefore hands the whole backlog to a
    single worker. ARC-Challenge filled 293 records that way on one GPU while
    three sat idle. Sharding over the outstanding pairs spreads them instead.
    """
    pairs = [(item, direction) for item in range(40) for direction in ("A_to_B", "B_to_A")]

    # What a world-8 worker owned, and therefore what it left behind.
    backlog = pairs[7::8]

    # The trap: every smaller world that divides 8 keeps the backlog together.
    for world in (2, 4):
        owners = {
            rank for rank in range(world)
            for pair in pairs[rank::world] if pair in set(backlog)
        }
        assert len(owners) == 1, f"world {world} spread the backlog unexpectedly"

    # Sharding over the outstanding pairs spreads them across every worker.
    for world in (2, 4):
        shards = [backlog[rank::world] for rank in range(world)]
        assert all(shards), f"world {world} left a worker idle"
        assert sorted(p for s in shards for p in s) == sorted(backlog)
        assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_marker_appears_exactly_once_and_only_for_latent_conditions():
    assert _revision("medqa", "true_statebridge").count(EMBEDDING_HINT_MARKER) == 1
    assert EMBEDDING_HINT_MARKER not in _revision("medqa", "none")
    assert EMBEDDING_HINT_MARKER not in _revision(
        "medqa", "true_text", sender_reasoning="<<SENDER>>"
    )


def test_external_block_rejects_unknown_conditions_and_missing_text():
    with pytest.raises(ValueError):
        external_block("true_telepathy")
    with pytest.raises(ValueError):
        external_block("true_text")
