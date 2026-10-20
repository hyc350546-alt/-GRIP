"""Dependency-free Pool60 gate with main and eligible-mix mean filters."""

import math


def select_slots(main, aux):
    if len(main) != 60 or len(aux) != 60:
        raise ValueError("Pool60 has exactly 60 slots, including cross-branch duplicates")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [*main, *aux]):
        raise ValueError("Scores must be finite and in [0,1]")
    mean = sum(main) / 60
    eligible = [i for i, value in enumerate(main) if value > mean]
    mixed = {i: 0.5 * main[i] + 0.5 * aux[i] for i in eligible}
    best = max(mixed.values(), default=0)
    # Center the mean so tied scores remain exactly equal to it.
    mix_mean = (
        best + math.fsum(value - best for value in mixed.values()) / len(mixed)
        if mixed else None
    )
    keep = [
        i for i in eligible
        if mixed[i] > mix_mean and best - mixed[i] <= 0.06 + 1e-12
    ]
    return {"main_mean": mean, "mix_mean": mix_mean, "eligible": eligible,
            "best": best, "keep": keep, "active": bool(keep)}


def gate_weights(main, aux):
    import torch
    result = select_slots(main, aux)
    weights = torch.zeros(60, dtype=torch.float32)
    if result["keep"]:
        logits = torch.tensor(
            [0.5 * main[i] + 0.5 * aux[i] for i in result["keep"]], dtype=torch.float32)
        weights[result["keep"]] = torch.softmax((logits - logits.max()) / 0.05, dim=0)
    return weights, result
