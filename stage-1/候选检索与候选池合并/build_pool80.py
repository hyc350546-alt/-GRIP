"""Build the recovered CIRCO V40+T40 candidate cache without synthetic images."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.circo_workflow import (
    external_workdir,
    load_queries,
    load_stage,
    normalized_id,
    require_new,
    write_jsonl,
)


def build_pool80(queries, index_path, source_path, cache_dir, output, device="cpu"):
    import torch
    import torch.nn.functional as F

    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    output = Path(output)
    manifest = output / "pool80.manifest.jsonl"
    aug_path = output / "aug_features_L.npz"
    require_new([manifest, aug_path])
    index = load(index_path)
    names = [normalized_id(n) for n in index["name"]]
    raw = index["feature"].float()
    if len(set(names)) != len(names) or raw.shape != (len(names), 768):
        raise ValueError(
            "Expected unique CIRCO gallery IDs with [N,768] ViT-L features"
        )
    source = load(source_path)
    cache = Path(cache_dir)
    target = torch.stack(
        [load(cache / f"gpt_predicted_features_{i}.pt") for i in range(15)]
    )
    before = torch.stack(
        [load(cache / f"blip_predicted_features_{i}.pt") for i in range(15)]
    )
    if target.shape != (15, len(queries), 768) or before.shape != target.shape:
        raise ValueError("Caption cache/query alignment mismatch")
    text_builder = load_stage("stage-1/双视角查询构建/T视角查询/build_text_query.py")
    visual_builder = load_stage(
        "stage-1/双视角查询构建/V视角查询/build_visual_query.py"
    )
    qt = text_builder.build_text_query(target)
    qb = text_builder.build_text_query(before)
    sources = torch.stack(
        [
            torch.as_tensor(source[normalized_id(r["reference_img_id"])])
            .float()
            .reshape(-1, 768)
            .mean(0)
            for r in queries
        ]
    )
    qv = visual_builder.build_visual_query(sources, qb, qt)
    gallery = F.normalize(raw, dim=-1).to(device)
    by_name = {n: i for i, n in enumerate(names)}
    rows, combined = [], {}
    for start in range(0, len(queries), 16):
        visual = (
            (qv[start : start + 16].to(device) @ gallery.T)
            .topk(41, dim=-1)
            .indices.cpu()
        )
        textual = (
            (qt[start : start + 16].to(device) @ gallery.T)
            .topk(40, dim=-1)
            .indices.cpu()
        )
        for j in range(len(visual)):
            row = queries[start + j]
            ref = normalized_id(row["reference_img_id"])
            v_ids = [names[i] for i in visual[j].tolist() if names[i] != ref][:40]
            t_ids = [names[i] for i in textual[j].tolist()]
            ids = v_ids + t_ids
            features = raw[[by_name[n] for n in ids]].unsqueeze(0).contiguous()
            # The historical final script accepts reference-keyed feature caches.
            if ref in combined and not torch.equal(combined[ref], features):
                raise ValueError(
                    f"Reference {ref} has conflicting query-specific candidates"
                )
            combined[ref] = features
            rows.append(
                {
                    "query_id": row["query_id"],
                    "reference_img_id": ref,
                    "relative_caption": row["relative_caption"],
                    "shared_concept": row["shared_concept"],
                    "target_captions": row["target_captions"],
                    "before_captions": row["before_captions"],
                    "candidate_ids": ids,
                    "candidate_sources": ["robust_visual"] * 40 + ["text"] * 40,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"combined": combined, "txt": {}}, aug_path)
    write_jsonl(manifest, rows)
    print(f"Saved {len(rows)} V40+T40 queries to {output}", flush=True)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    return build_pool80(
        load_queries(),
        args.index,
        args.source,
        args.cache_dir,
        external_workdir(args.output),
        args.device,
    )


if __name__ == "__main__":
    main()
