"""Independent -> Communicate -> Revise evaluation framework."""

PROTOCOL = "ICR-MEDQA300-V2"  # legacy V2 runs
PROTOCOL_V3 = "ICR-V3"
CONDITIONS = (
    "none",
    # Channel x message source. "true" sends the actual sender's message; "self"
    # (the receiver's own trajectory) and "other" (the same sender on another
    # item) are content controls.
    "true_text",
    "self_text",
    "other_text",
    "true_answer",
    "self_answer",
    "other_answer",
    "true_statebridge",
    "self_statebridge",
    "other_statebridge",
    "true_latentmas",
    "self_latentmas",
    "other_latentmas",
)
# A third independent sample per item. The pairs are what the audit measures,
# and CR and PR only exist on pairs whose two agents disagree about
# correctness, so a third agent is far cheaper than a third replication: it
# costs 50% more beliefs but turns 2 ordered pairs per item into 6, and any
# item with mixed correctness now yields 2 correction and 2 destruction cases
# instead of 1 each.
AGENTS = ("A", "B", "C")
DIRECTIONS = tuple(
    f"{sender}_to_{receiver}"
    for sender in AGENTS
    for receiver in AGENTS
    if sender != receiver
)


def direction_agents(direction: str) -> tuple[str, str]:
    """Sender and receiver of a direction label."""
    sender, _, receiver = direction.partition("_to_")
    if sender not in AGENTS or receiver not in AGENTS or sender == receiver:
        raise ValueError(f"Unknown direction: {direction}")
    return sender, receiver
