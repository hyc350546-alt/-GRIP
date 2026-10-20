"""CIRCO semantic: complete implementation with a direct CLI."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.circo_workflow import ROOT, render_prompt, external_workdir, require_new


def run(cache_dir, index, output, batch_size=32):
    import argparse
    import torch
    import torch.nn.functional as F

    args = argparse.Namespace(
        cache_dir=cache_dir, index=index, output=output, batch_size=batch_size
    )
    temperature = 0.01
    external_workdir(Path(output).parent)
    require_new([output])
    cache_dir = Path(args.cache_dir)
    gpt, blip = ([], [])
    for i in range(15):
        gp = cache_dir / f"gpt_predicted_features_{i}.pt"
        bp = cache_dir / f"blip_predicted_features_{i}.pt"
        g = torch.load(gp, map_location="cpu", weights_only=False).float()
        b = torch.load(bp, map_location="cpu", weights_only=False).float()
        if tuple(g.shape) != (800, 768) or tuple(b.shape) != (800, 768):
            raise RuntimeError(
                f"caption={i}: GPT={tuple(g.shape)}, BLIP={tuple(b.shape)}"
            )
        gpt.append(g)
        blip.append(b)
    index_pack = torch.load(args.index, map_location="cpu", weights_only=False)
    gallery = F.normalize(index_pack["feature"].float(), dim=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery = gallery.to(device)
    penalties = []
    for i, (g, b) in enumerate(zip(gpt, blip), 1):
        total = 0.0
        for start in range(0, 800, args.batch_size):
            end = min(start + args.batch_size, 800)
            after = g[start:end].to(device) @ gallery.T
            before = b[start:end].to(device) @ gallery.T
            penalty = torch.topk(torch.relu(before - after), k=50, dim=-1).values.sum()
            total += float(penalty.item())
        penalties.append(total)
        print(f"caption {i}/15 penalty={total:.6f}", flush=True)
    penalties = torch.tensor(penalties, dtype=torch.float32)
    debiased_weight = torch.softmax(penalties / penalties.max() / temperature, dim=0)
    gpt_stack = torch.stack(gpt)
    blip_stack = torch.stack(blip)
    f_t = torch.sum(gpt_stack * debiased_weight[:, None, None], dim=0)
    semantic_direction = torch.mean(gpt_stack - blip_stack, dim=0)
    output = {
        "f_t": f_t.cpu(),
        "semantic_direction": semantic_direction.cpu(),
        "debiased_weight": debiased_weight.cpu(),
        "meta": {
            "source": "15 correct CIRCO ViT-L GPT/BLIP caches",
            "debiased_temperature": temperature,
            "queries": 800,
            "feature_dim": 768,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print("saved:", args.output)
    print("f_t:", tuple(f_t.shape))
    print("semantic_direction:", tuple(semantic_direction.shape))
    print("debiased weights:", debiased_weight.tolist())


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return run(args.cache_dir, args.index, args.output, args.batch_size)


if __name__ == "__main__":
    main()
