"""Candidate T query, distinct from the final debiased semantic f_t."""


def build_text_query(caption_features):
    import torch.nn.functional as F
    if caption_features.ndim != 3 or caption_features.shape[0] != 15:
        raise ValueError("Expected already-normalized caption features [15, queries, dim]")
    return F.normalize(caption_features.float().mean(dim=0), dim=-1)
