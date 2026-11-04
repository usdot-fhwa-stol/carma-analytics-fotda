"""Greedy forward pairing of log events that carry no correlatable content.

Most stages in this pipeline can be joined on content (the detection's
``objectId``/``timestamp``, or the ASN.1-UPER payload bytes). A few log lines
carry neither -- ``TenaV2Xplugin.cpp (163)``, ``SendMessage.cpp (28)`` and
``ImmediateForwardPlugin.cpp (326)`` log only that *a* message moved. Those are
emitted synchronously by the same thread that just handled the identified
message, so pairing each one to the nearest preceding identified event is sound
as long as we bound the gap and refuse to reuse an event.
"""

from __future__ import annotations


def pair_forward(
    anchor_ts: list[float],
    follower_ts: list[float],
    max_gap_ms: float = 250.0,
    back_tolerance_ms: float = 2.0,
) -> list[int | None]:
    """Match each anchor to the first unused follower at/after it.

    ``back_tolerance_ms`` allows a follower to appear marginally *before* its
    anchor, which happens when two log lines share a millisecond or when the
    stamp written into a message header predates the line that logs it.

    Returns a list parallel to ``anchor_ts`` holding the matched index into
    ``follower_ts``, or None where no follower fell inside the window.
    """
    if not anchor_ts or not follower_ts:
        return [None] * len(anchor_ts)

    out: list[int | None] = []
    j = 0
    n = len(follower_ts)
    for a in anchor_ts:
        while j < n and follower_ts[j] < a - back_tolerance_ms:
            j += 1
        if j < n and follower_ts[j] <= a + max_gap_ms:
            out.append(j)
            j += 1
        else:
            out.append(None)
    return out
