"""The selected-slot weighted proxy. Do not renormalize the resulting sum."""


def aggregate_proxy(candidate_features, weights):
    import torch.nn.functional as F
    if candidate_features.ndim != 2 or candidate_features.shape[0] != len(weights):
        raise ValueError("Candidate features and slot weights are not aligned")
    return (F.normalize(candidate_features.float(), dim=-1) * weights[:, None]).sum(dim=0)
