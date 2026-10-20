"""No-synthetic-proxy V query; ordered Top40 verified on eight cloud queries."""


def build_visual_query(source, before_text, target_text):
    import torch.nn.functional as F
    source = F.normalize(source.float(), dim=-1)
    direction = F.normalize(target_text - before_text, dim=-1)
    robust = F.normalize(source + direction, dim=-1)
    return F.normalize(0.8 * robust + 0.2 * target_text, dim=-1)
