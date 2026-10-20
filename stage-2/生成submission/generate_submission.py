"""CIRCO submission: complete implementation with a direct CLI."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.circo_workflow import ROOT, render_prompt, external_workdir, require_new
import json
import math
import os
from collections import Counter
from pathlib import Path

VISUAL_K = 30
TEXT_K = 30
POOL_K = VISUAL_K + TEXT_K
MAIN_WEIGHT = 0.5
AUX_WEIGHT = 0.5
DELTA = 0.06
TAU = 0.05
ACTIVE_T = 1.0
ACTIVE_A = 12.0


def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def norm_id(value):
    text = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path}, line {line_no}: {exc}"
                ) from exc
    return rows


def load_scores(path, field):
    scores = {}
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("query_id") is None or row.get("image_id") is None:
            raise RuntimeError(f"{path}:{line_no} lacks query_id/image_id")
        key = (str(row["query_id"]), norm_id(row["image_id"]))
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{path}:{line_no} invalid {field}: {row.get(field)!r}"
            ) from exc
        if not math.isfinite(value):
            raise RuntimeError(
                f"{path}:{line_no} non-finite {field} for {key}: {value}"
            )
        if key in scores and (
            not math.isclose(scores[key], value, rel_tol=0.0, abs_tol=1e-07)
        ):
            raise RuntimeError(f"Conflicting duplicate score for {key} in {path}")
        scores[key] = value
    return scores


def branch_from_source(source):
    text = str(source).lower()
    if "text" in text:
        return "text"
    if any((token in text for token in ("visual", "vision", "robust", "image"))):
        return "visual"
    return None


def resolve_feature_key(combined, key_by_text, row):
    if torch.is_tensor(combined):
        return None
    possible = [
        row.get("query_id"),
        row.get("reference_img_id"),
        row.get("reference_name"),
    ]
    for value in list(possible):
        if value is not None:
            possible.append(norm_id(value))
    for value in possible:
        if value is None:
            continue
        text = str(value)
        if text in key_by_text:
            return key_by_text[text]
    raise RuntimeError(
        "Cannot map query {} to aug feature key. Tried query/reference keys: {}".format(
            row.get("query_id"), possible
        )
    )


def run(paths, output):
    global torch, F, tqdm
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    OUT = external_workdir(output)
    require_new([OUT])
    MANIFEST = Path(paths["manifest"])
    AUG80 = Path(paths["aug80"])
    INDEX = Path(paths["index"])
    MAIN = Path(paths["main"])
    AUX = Path(paths["aux"])
    SEMANTIC = Path(paths["semantic"])
    GATE = OUT / "gate.pt"
    POOL60_MANIFEST = OUT / "pool60.manifest.jsonl"
    SUBMISSION = OUT / "submission.json"
    REPORT = OUT / "report.json"
    OUT.mkdir(parents=True, exist_ok=True)
    rows80 = load_jsonl(MANIFEST)
    query_ids = [str(row.get("query_id")) for row in rows80]
    if len(rows80) != 800 or query_ids != [str(i) for i in range(800)]:
        raise RuntimeError(
            "Expected CIRCO queries exactly 0..799 in manifest order, got {} rows; first={}".format(
                len(rows80), query_ids[:5]
            )
        )
    aug_pack = tload(AUG80)
    if not isinstance(aug_pack, dict) or "combined" not in aug_pack:
        raise RuntimeError(f"Unexpected aug pack: {AUG80}")
    combined = aug_pack["combined"]
    if torch.is_tensor(combined):
        key_by_text = {}
    elif isinstance(combined, dict):
        key_by_text = {str(key): key for key in combined.keys()}
    else:
        raise RuntimeError(f"Unexpected combined type: {type(combined)}")
    rows60 = []
    pool_positions = []
    feature_keys = []
    source_counts = Counter()
    for row in rows80:
        candidate_ids = list(row.get("candidate_ids", []))
        candidate_sources = row.get("candidate_sources")
        if len(candidate_ids) != 80:
            raise RuntimeError(
                f"Query {row['query_id']} has {len(candidate_ids)} candidate slots; expected Pool80."
            )
        if not isinstance(candidate_sources, list) or len(candidate_sources) != 80:
            raise RuntimeError(
                f"Query {row['query_id']} lacks 80 aligned candidate_sources; cannot safely build Pool60."
            )
        visual_positions = []
        text_positions = []
        unknown_sources = []
        for index, source in enumerate(candidate_sources):
            source_counts[str(source)] += 1
            branch = branch_from_source(source)
            if branch == "visual":
                visual_positions.append(index)
            elif branch == "text":
                text_positions.append(index)
            else:
                unknown_sources.append(str(source))
        if unknown_sources:
            raise RuntimeError(
                f"Query {row['query_id']} has unknown source labels: {sorted(set(unknown_sources))}"
            )
        if len(visual_positions) < VISUAL_K or len(text_positions) < TEXT_K:
            raise RuntimeError(
                "Query {} has visual/text counts {}/{}; need at least {}/{}.".format(
                    row["query_id"],
                    len(visual_positions),
                    len(text_positions),
                    VISUAL_K,
                    TEXT_K,
                )
            )
        selected = visual_positions[:VISUAL_K] + text_positions[:TEXT_K]
        if len(selected) != POOL_K:
            raise RuntimeError(f"Pool60 selection failed for query {row['query_id']}")
        reduced = dict(row)
        reduced["candidate_ids"] = [candidate_ids[index] for index in selected]
        reduced["candidate_sources"] = [candidate_sources[index] for index in selected]
        reduced["pool80_positions"] = selected
        for key in ("candidate_paths", "candidate_ranks"):
            values = row.get(key)
            if isinstance(values, list) and len(values) == 80:
                reduced[key] = [values[index] for index in selected]
        rows60.append(reduced)
        pool_positions.append(selected)
        feature_keys.append(resolve_feature_key(combined, key_by_text, row))
    write_jsonl(POOL60_MANIFEST, rows60)
    main_scores = load_scores(MAIN, "main")
    aux_scores = load_scores(AUX, "aux")
    weights = torch.zeros((len(rows60), POOL_K), dtype=torch.float32)
    positions_tensor = torch.tensor(pool_positions, dtype=torch.long)
    main_means = []
    selected_counts = []
    fallback_no_eligible = 0
    fallback_no_mix_gt_mean = 0
    main_gt_mean_slots = 0
    for query_index, row in enumerate(rows60):
        query_id = str(row["query_id"])
        main_values = []
        aux_values = []
        for image_id in row["candidate_ids"]:
            key = (query_id, norm_id(image_id))
            if key not in main_scores or key not in aux_scores:
                raise RuntimeError(f"Missing main/aux score for Pool60 pair {key}")
            main_values.append(main_scores[key])
            aux_values.append(aux_scores[key])
        main_mean = sum(main_values) / POOL_K
        main_means.append(main_mean)
        eligible = [
            index for index, value in enumerate(main_values) if value > main_mean
        ]
        main_gt_mean_slots += len(eligible)
        if not eligible:
            fallback_no_eligible += 1
            selected_counts.append(0)
            continue
        mixed = {
            index: MAIN_WEIGHT * main_values[index] + AUX_WEIGHT * aux_values[index]
            for index in eligible
        }
        best = max(mixed.values())
        # Center the mean so tied scores remain exactly equal to it.
        mix_mean = best + math.fsum(value - best for value in mixed.values()) / len(mixed)
        keep = [
            index for index in eligible
            if mixed[index] > mix_mean and best - mixed[index] <= DELTA + 1e-12
        ]
        if not keep:
            fallback_no_mix_gt_mean += 1
            selected_counts.append(0)
            continue
        logits = torch.tensor([mixed[index] for index in keep], dtype=torch.float32)
        probabilities = torch.softmax((logits - logits.max()) / TAU, dim=0)
        for index, probability in zip(keep, probabilities):
            weights[query_index, index] = probability
        selected_counts.append(len(keep))
    proxy_gate = (weights.sum(dim=1) > 0).float()
    active_counts = [count for count in selected_counts if count > 0]
    gate_pack = {
        "weights": weights,
        "proxy_gate": proxy_gate,
        "pool_positions": positions_tensor,
        "selected_counts": selected_counts,
        "main_means": torch.tensor(main_means, dtype=torch.float32),
        "meta": {
            "scorer": "Qwen3.7-plus",
            "pool": "Pool60 = first 30 visual + first 30 text slots from original Pool80",
            "eligibility": "strict main(candidate) > mean main(over the 60 selected Pool60 slots)",
            "score_after_main_filter": "0.5*main + 0.5*aux",
            "mix_filter": "strict mix > mean mix(over main-eligible Pool60 slots)",
            "delta": DELTA,
            "temperature": TAU,
            "active_ratio": "s:t:a=0:1:12",
            "fallback_ratio": "s:t:a=0:1:0",
            "final_similarity": "St*Sp",
        },
    }
    gate_tmp = GATE.with_name(GATE.name + ".tmp")
    torch.save(gate_pack, gate_tmp)
    os.replace(gate_tmp, GATE)
    gate_audit = {
        "queries": len(rows60),
        "pool_size": POOL_K,
        "pool_definition": "first 30 visual + first 30 text slots from original Pool80",
        "score": "0.5*main + 0.5*aux",
        "main_filter": "strict main > per-query Pool60 main mean",
        "mix_filter": "strict mix > mean mix(over main-eligible Pool60 slots)",
        "delta": DELTA,
        "temperature": TAU,
        "active_ratio": "0:1:12",
        "fallback_ratio": "0:1:0",
        "mean_of_per_query_main_means": sum(main_means) / len(main_means),
        "main_gt_pool60_mean_candidate_slots": main_gt_mean_slots,
        "main_gt_pool60_mean_candidate_slot_percent": 100.0
        * main_gt_mean_slots
        / (len(rows60) * POOL_K),
        "active_gate_queries": int(proxy_gate.sum().item()),
        "fallback_queries": int((proxy_gate == 0).sum().item()),
        "fallback_no_main_gt_pool60_mean": fallback_no_eligible,
        "fallback_no_mix_gt_eligible_mean": fallback_no_mix_gt_mean,
        "selected_positions": int((weights > 0).sum().item()),
        "active_selected_min": min(active_counts) if active_counts else 0,
        "active_selected_mean": (
            sum(active_counts) / len(active_counts) if active_counts else 0.0
        ),
        "active_selected_max": max(active_counts) if active_counts else 0,
        "source_label_counts_in_original_pool80": dict(sorted(source_counts.items())),
        "main_score_file": str(MAIN),
        "aux_score_file": str(AUX),
        "manifest": str(MANIFEST),
    }
    write_json(OUT / "gate_audit.json", gate_audit)
    write_json(
        OUT / "pool60_selection.json",
        {
            "pool_definition": gate_audit["pool_definition"],
            "queries": len(rows60),
            "pool_size": POOL_K,
            "manifest": str(POOL60_MANIFEST),
        },
    )
    semantic = tload(SEMANTIC)
    if not isinstance(semantic, dict):
        raise RuntimeError(f"Unexpected semantic cache: {SEMANTIC}")
    f_t = torch.as_tensor(semantic["f_t"]).float()
    direction = torch.as_tensor(semantic["semantic_direction"]).float()
    if tuple(f_t.shape) != (800, 768):
        raise RuntimeError(f"Expected cached f_t [800,768], got {tuple(f_t.shape)}")
    if tuple(direction.shape) != (800, 768):
        raise RuntimeError(
            f"Expected cached semantic_direction [800,768], got {tuple(direction.shape)}"
        )
    index_pack = tload(INDEX)
    gallery = torch.as_tensor(index_pack["feature"]).float()
    index_names = [str(item) for item in index_pack["name"]]
    if gallery.ndim != 2 or gallery.shape[1] != 768:
        raise RuntimeError(f"Unexpected gallery shape: {tuple(gallery.shape)}")
    if len(index_names) != gallery.shape[0]:
        raise RuntimeError("Gallery feature/name count mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery = F.normalize(gallery, dim=-1).to(device)
    f_t = F.normalize(f_t, dim=-1).to(device)
    direction = direction.to(device)
    weights = weights.to(device)
    proxy_gate = proxy_gate.to(device)
    positions_tensor = positions_tensor.to(device)
    robust_rows = []
    for query_index, row in enumerate(rows80):
        if torch.is_tensor(combined):
            raw_candidates = combined[query_index]
        else:
            raw_candidates = combined[feature_keys[query_index]]
        raw_candidates = torch.as_tensor(raw_candidates).float().squeeze(0)
        if tuple(raw_candidates.shape) != (80, 768):
            raise RuntimeError(
                "Query {} has aug feature shape {}, expected [80,768].".format(
                    row["query_id"], tuple(raw_candidates.shape)
                )
            )
        candidates = raw_candidates[positions_tensor[query_index].cpu()]
        candidates = F.normalize(candidates, dim=-1).to(device)
        proxy = (candidates * weights[query_index, :, None]).sum(dim=0)
        query_direction = direction[query_index]
        direction_norm = query_direction.norm()
        proxy_norm = proxy.norm()
        active_sp = ACTIVE_T * (proxy_norm / (direction_norm + 1e-08)) * query_direction
        active_sp = active_sp + ACTIVE_A * proxy
        fallback_sp = query_direction / (direction_norm + 1e-08)
        if proxy_gate[query_index] > 0:
            robust_rows.append(active_sp)
        else:
            robust_rows.append(fallback_sp)
    robust = torch.stack(robust_rows, dim=0)
    predictions = {}
    batch_size = 16
    for start in tqdm(range(0, len(query_ids), batch_size), desc="retrieve CIRCO"):
        end = min(start + batch_size, len(query_ids))
        st = f_t[start:end] @ gallery.T
        sp = robust[start:end] @ gallery.T
        score = st * sp
        if not torch.isfinite(score).all():
            raise RuntimeError(
                f"Non-finite final similarity at query range {start}:{end}"
            )
        top_indices = torch.topk(score, k=50, dim=-1).indices.cpu().tolist()
        for offset, ranked in enumerate(top_indices):
            predictions[query_ids[start + offset]] = [
                index_names[index] for index in ranked
            ]
    if len(predictions) != 800 or any(
        (len(value) != 50 for value in predictions.values())
    ):
        raise RuntimeError("Submission validation failed")
    write_json(SUBMISSION, predictions)
    report = {
        "submission": str(SUBMISSION),
        "queries": len(predictions),
        "topk": 50,
        "backbone": "OpenAI CLIP ViT-L/14",
        "feature_dim": 768,
        "pool_size": POOL_K,
        "pool_definition": gate_audit["pool_definition"],
        "active_gate_queries": gate_audit["active_gate_queries"],
        "fallback_queries": gate_audit["fallback_queries"],
        "gate": {
            "main_weight": MAIN_WEIGHT,
            "aux_weight": AUX_WEIGHT,
            "main_filter": gate_audit["main_filter"],
            "mix_filter": gate_audit["mix_filter"],
            "delta": DELTA,
            "temperature": TAU,
        },
        "retrieval": {
            "active_ratio": "0:1:12",
            "fallback_ratio": "0:1:0",
            "formula": "St*Sp",
            "active_sp": "(||proxy||/||direction||)*direction + 12*proxy",
            "fallback_sp": "direction/||direction||",
        },
        "gate_path": str(GATE),
        "pool60_manifest": str(POOL60_MANIFEST),
        "semantic_cache": str(SEMANTIC),
    }
    write_json(REPORT, report)
    print(json.dumps(gate_audit, indent=2))
    print(json.dumps(report, indent=2))
    print("saved submission:", SUBMISSION)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "aug80", "index", "main", "aux", "semantic"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    paths = {
        name: getattr(args, name)
        for name in ("manifest", "aug80", "index", "main", "aux", "semantic")
    }
    for name, filename in paths.items():
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Missing {name}: {filename}")
    return run(paths, args.output)


if __name__ == "__main__":
    main()
