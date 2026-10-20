"""CIRCO aux: complete implementation with a direct CLI."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.circo_workflow import ROOT, render_prompt, external_workdir, require_new
import argparse
import base64
import io
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from PIL import Image


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm_id(value):
    value = str(value)
    for suffix in (".jpg", ".jpeg", ".png"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
            break
    return str(int(value)) if value.isdigit() else value


def find_image(image_root, image_id):
    image_id = norm_id(image_id)
    candidates = []
    if image_id.isdigit():
        candidates.append(image_root / f"{int(image_id):012d}.jpg")
    candidates.extend(
        (image_root / f"{image_id}{suffix}" for suffix in (".jpg", ".png", ".jpeg"))
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found: {image_id} under {image_root}")


def resolve_manifest_image(row, image_id, image_root=None, reference=False):
    """Resolve both CIRCO ids and CIRR paths without changing score semantics."""
    if reference and row.get("reference_path"):
        path = Path(row["reference_path"])
        if path.exists():
            return path
    wanted = norm_id(image_id)
    for candidate_id, candidate_path in zip(
        row.get("candidate_ids", []), row.get("candidate_paths", [])
    ):
        if norm_id(candidate_id) == wanted:
            path = Path(candidate_path)
            if path.exists():
                return path
    if image_root:
        return find_image(Path(image_root), wanted)
    raise FileNotFoundError(
        f"Image {wanted} is absent from manifest paths and --image-root was not set"
    )


@lru_cache(maxsize=4096)
def image_data_url(path_string, max_side, quality):
    with Image.open(path_string) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_json_response(text):
    text = text.strip()
    fenced = re.search("```(?:json)?\\s*([\\s\\S]*?)```", text, flags=re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search("\\{[\\s\\S]*\\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_score(value):
    value = float(value)
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def geometric_mean(values, eps=1e-06):
    values = [max(eps, min(1.0 - eps, float(x))) for x in values]
    if not values:
        return 0.5
    return math.exp(sum((math.log(x) for x in values)) / len(values))


def load_question_bank(path):
    bank = {}
    seen = {}
    for row in load_jsonl(path):
        qid = str(row["query_id"])
        bank.setdefault(qid, [])
        seen.setdefault(qid, set())
        for item in row.get("questions", []):
            if item.get("valid", True) is False or not item.get("question"):
                continue
            question = {
                "question": str(item["question"]).strip(),
                "polarity": str(item.get("polarity", "positive")),
                "use_reference": bool(item.get("use_reference", False)),
                "concept": item.get("concept", {}),
            }
            key = (
                question["question"],
                question["polarity"],
                question["use_reference"],
            )
            if key not in seen[qid]:
                seen[qid].add(key)
                bank[qid].append(question)
    return bank


def existing_keys(path):
    return {
        (str(row["query_id"]), norm_id(row["image_id"])) for row in load_jsonl(path)
    }


def call_api(args, reference_path, candidates, questions):
    question_lines = "\n".join(
        (f"Q{index}: {item['question']}" for index, item in enumerate(questions, 1))
    )
    candidate_lines = "\n".join(
        (
            f"Candidate {index}: Image {index + 1}"
            for index in range(1, len(candidates) + 1)
        )
    )
    prompt = render_prompt(
        "aux", [f"{len(candidates) + 1}", f"{candidate_lines}", f"{question_lines}"]
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Reference image (Image 1):"},
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url(
                    str(reference_path), args.max_image_side, args.jpeg_quality
                )
            },
        },
    ]
    for candidate_index, candidate in enumerate(candidates, 1):
        content.append(
            {
                "type": "text",
                "text": f"Candidate {candidate_index} (Image {candidate_index + 1}):",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url(
                        str(candidate["path"]), args.max_image_side, args.jpeg_quality
                    )
                },
            }
        )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": 0,
        "enable_thinking": False,
    }
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(args.retries):
        try:
            response = requests.post(
                args.base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=args.timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:1000]}"
                )
            body = response.json()
            parsed = parse_json_response(body["choices"][0]["message"]["content"])
            by_candidate = {
                int(item["candidate_index"]): item for item in parsed["scores"]
            }
            expected_candidates = set(range(1, len(candidates) + 1))
            if set(by_candidate) != expected_candidates:
                raise ValueError(
                    f"Unexpected candidate indexes: {sorted(by_candidate)}"
                )
            output = []
            for candidate_index in range(1, len(candidates) + 1):
                answers = {
                    int(item["question_index"]): item
                    for item in by_candidate[candidate_index]["answers"]
                }
                expected_questions = set(range(1, len(questions) + 1))
                if set(answers) != expected_questions:
                    raise ValueError(f"Unexpected question indexes: {sorted(answers)}")
                output.append(
                    [
                        clamp_score(answers[index]["yes_probability"])
                        for index in range(1, len(questions) + 1)
                    ]
                )
            return output
        except Exception as exc:
            last_error = exc
            if attempt + 1 < args.retries:
                time.sleep(min(30.0, 2**attempt + random.random()))
    raise RuntimeError(
        f"API request failed after {args.retries} attempts: {last_error}"
    )


def score_batch(args, row, questions, candidate_ids):
    reference_id = norm_id(row.get("reference_img_id", row.get("reference_name", "")))
    reference_path = resolve_manifest_image(
        row, reference_id, args.image_root, reference=True
    )
    candidates = [
        {"id": image_id, "path": resolve_manifest_image(row, image_id, args.image_root)}
        for image_id in candidate_ids
    ]
    yes_probabilities = call_api(args, reference_path, candidates, questions)
    outputs = []
    for candidate, probabilities in zip(candidates, yes_probabilities):
        details = []
        for question, p_yes in zip(questions, probabilities):
            score = 1.0 - p_yes if question["polarity"] == "negative" else p_yes
            details.append({**question, "p_yes": p_yes, "score": score})
        outputs.append(
            {
                "query_id": str(row["query_id"]),
                "reference_img_id": reference_id,
                "image_id": candidate["id"],
                "aux": geometric_mean((item["score"] for item in details)),
                "questions": details,
                "scorer": args.model,
                "verifier_prompt": "strict-atomic-v3-labeled-images",
            }
        )
    return outputs


def score_batch_resilient(args, row, questions, candidate_ids):
    try:
        return score_batch(args, row, questions, candidate_ids)
    except Exception:
        if len(candidate_ids) <= 1:
            raise
        middle = len(candidate_ids) // 2
        return score_batch_resilient(
            args, row, questions, candidate_ids[:middle]
        ) + score_batch_resilient(args, row, questions, candidate_ids[middle:])


def parse_args(argv=None, api_key=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--question-scores", required=True)
    parser.add_argument(
        "--image-root",
        default="",
        help="Optional for CIRR manifests that already contain reference/candidate paths.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-url",
        default="https://ws-0t5g9c985o5ikzqu.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY"))
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--batch-candidates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-query", type=int, default=0)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--skip-failed-batches",
        action="store_true",
        help="Leave failed candidates missing so a later retry can resume them.",
    )
    args = parser.parse_args(argv)
    if api_key is not None:
        args.api_key = api_key
    elif not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY")
    return args


def run_pipeline(args):
    rows = load_jsonl(args.manifest)
    question_bank = load_question_bank(args.question_scores)
    end = len(rows)
    if args.limit_queries is not None:
        end = min(end, args.start_query + args.limit_queries)
    selected_rows = rows[args.start_query : end]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(output)
    expected_keys = {
        (str(row["query_id"]), norm_id(image_id))
        for row in selected_rows
        for image_id in row.get("candidate_ids", [])
    }
    covered_keys = expected_keys & done
    missing_keys = expected_keys - done
    question_counts = [len(question_bank.get(str(row["query_id"]), [])) for row in rows]
    print("queries selected:", len(selected_rows))
    print("question-bank queries:", len(question_bank))
    print(
        "questions per query min/mean/max:",
        min(question_counts),
        sum(question_counts) / len(question_counts),
        max(question_counts),
    )
    print("existing unique candidate scores:", len(done))
    print("selected candidate slots:", len(expected_keys))
    print("covered selected slots:", len(covered_keys))
    print("missing selected slots:", len(missing_keys))
    if missing_keys:
        print("missing examples:", sorted(missing_keys)[:10])
    shown = 0
    for row in rows:
        qid = str(row["query_id"])
        questions = question_bank.get(qid, [])
        if not questions:
            continue
        print(f"\nquery {qid} question examples:")
        for item in questions:
            print(
                " -",
                item["polarity"],
                "reference" if item["use_reference"] else "candidate",
                item["question"],
            )
        shown += 1
        if shown == 3:
            break
    if args.audit_only:
        return
    if not args.api_key:
        raise ValueError("Set DASHSCOPE_API_KEY or pass --api-key")
    written = 0
    failed = 0
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for query_offset, row in enumerate(selected_rows, args.start_query):
            qid = str(row["query_id"])
            questions = question_bank.get(qid, [])
            unique_ids = list(dict.fromkeys((norm_id(x) for x in row["candidate_ids"])))
            missing_ids = [
                image_id for image_id in unique_ids if (qid, image_id) not in done
            ]
            if not questions:
                for image_id in missing_ids:
                    result = {
                        "query_id": qid,
                        "reference_img_id": norm_id(
                            row.get("reference_img_id", row.get("reference_name", ""))
                        ),
                        "image_id": image_id,
                        "aux": 0.5,
                        "questions": [],
                        "scorer": "no-valid-questions-neutral",
                    }
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    done.add((qid, image_id))
                    written += 1
                continue
            tasks = []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for start in range(0, len(missing_ids), args.batch_candidates):
                    candidate_ids = missing_ids[start : start + args.batch_candidates]
                    tasks.append(
                        executor.submit(
                            score_batch_resilient, args, row, questions, candidate_ids
                        )
                    )
                for future in as_completed(tasks):
                    try:
                        results = future.result()
                    except Exception as exc:
                        failed += 1
                        print(f"WARNING query {qid}: batch skipped: {exc}", flush=True)
                        if not args.skip_failed_batches:
                            raise
                        continue
                    for result in results:
                        key = (qid, norm_id(result["image_id"]))
                        if key in done:
                            continue
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        done.add(key)
                        written += 1
            print(
                f"query {query_offset + 1}/{end}; unique={len(unique_ids)}; questions={len(questions)}; new rows={written}; failed batches={failed}",
                flush=True,
            )
    print("saved:", output)
    print("new unique candidate scores:", written)
    print("failed batches:", failed)


def main(argv=None, api_key=None):
    args = parse_args(argv, api_key)
    global requests
    import requests

    return run_pipeline(args)


def run(manifest, questions, images, output, base_url, workers, api_key):
    return main(
        [
            "--manifest",
            str(manifest),
            "--question-scores",
            str(questions),
            "--image-root",
            str(images),
            "--output",
            str(output),
            "--base-url",
            base_url,
            "--model",
            "qwen3.7-plus",
            "--workers",
            str(workers),
        ],
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
