"""CIRCO main: complete implementation with a direct CLI."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.circo_workflow import ROOT, render_prompt, external_workdir, require_new
import argparse
import base64
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from PIL import Image


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm_name(value):
    value = str(value)
    for suffix in (".jpg", ".jpeg", ".png"):
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]
    return value


def find_image(image_root, image_id, explicit_path=None):
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)
    stems = [str(image_id)]
    if str(image_id).isdigit():
        stems.append(f"{int(image_id):012d}")
    for stem in stems:
        for suffix in (".png", ".jpg", ".jpeg"):
            path = image_root / f"{stem}{suffix}"
            if path.exists():
                return path
    raise FileNotFoundError(f"Image not found: {image_id}")


@lru_cache(maxsize=4096)
def image_data_url(path_string, max_side, quality):
    path = Path(path_string)
    with Image.open(path) as image:
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


def call_api(
    session,
    url,
    api_key,
    model,
    reference_path,
    candidates,
    modification,
    target_caption,
    max_side,
    quality,
    timeout,
    retries,
):
    candidate_lines = "\n".join(
        (
            f"- Candidate {index}: Image {index + 1}"
            for index in range(1, len(candidates) + 1)
        )
    )
    prompt = render_prompt(
        "main",
        [
            f"{len(candidates) + 1}",
            f"{candidate_lines}",
            f"{modification}",
            f"{target_caption}",
        ],
    )
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url(str(reference_path), max_side, quality)
            },
        },
    ]
    for candidate in candidates:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url(str(candidate["path"]), max_side, quality)
                },
            }
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": 0,
        "enable_thinking": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(retries):
        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout)
            if not response.ok:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:2000]}"
                )
            body = response.json()
            message = body["choices"][0]["message"]["content"]
            parsed = parse_json_response(message)
            scores = parsed["scores"]
            by_index = {int(item["candidate_index"]): item for item in scores}
            if set(by_index) != set(range(1, len(candidates) + 1)):
                raise ValueError(f"Unexpected candidate indexes: {sorted(by_index)}")
            return [
                {
                    "main": clamp_score(by_index[index]["main_score"]),
                    "aux": clamp_score(by_index[index]["aux_score"]),
                }
                for index in range(1, len(candidates) + 1)
            ]
        except Exception as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            time.sleep(min(30.0, 2**attempt + random.random()))
    raise RuntimeError(f"API request failed after {retries} attempts: {last_error}")


def existing_keys(path):
    keys = set()
    if not path.exists():
        return keys
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((str(row["query_id"]), norm_name(row["image_id"])))
    return keys


def score_batch(
    args, row, caption_row, query_index, batch_start, batch_ids, batch_paths
):
    reference_id = norm_name(
        row.get("reference_name") or row.get("reference_img_id") or row.get("candidate")
    )
    reference_path = find_image(
        Path(args.image_root), reference_id, row.get("reference_path")
    )
    candidates = [
        {
            "id": image_id,
            "path": find_image(Path(args.image_root), image_id, explicit_path),
        }
        for image_id, explicit_path in zip(batch_ids, batch_paths)
    ]
    modification = (
        row.get("relative_caption")
        or " and ".join(row.get("captions", []))
        or " and ".join(caption_row.get("captions", []))
    )
    target_caption = (
        row.get("target_caption")
        or row.get("target_caption_original")
        or (row.get("target_captions") or [""])[0]
        or (row.get("multi_gpt-3.5_opt") or [""])[0]
        or caption_row.get("target_caption_original")
        or caption_row.get("target_caption")
        or (caption_row.get("multi_gpt-3.5_opt") or [""])[0]
    )
    with requests.Session() as session:
        scores = call_api(
            session=session,
            url=args.base_url.rstrip("/") + "/chat/completions",
            api_key=args.api_key,
            model=args.model,
            reference_path=reference_path,
            candidates=candidates,
            modification=modification,
            target_caption=target_caption,
            max_side=args.max_image_side,
            quality=args.jpeg_quality,
            timeout=args.timeout,
            retries=args.retries,
        )
    query_id = str(row.get("query_id", f"toptee_{query_index}"))
    outputs = []
    for offset, (candidate, score) in enumerate(zip(candidates, scores)):
        candidate_index = batch_start + offset
        outputs.append(
            {
                "query_id": query_id,
                "reference_name": reference_id,
                "relative_caption": modification,
                "candidate_index": candidate_index,
                "candidate_rank": candidate_index + 1,
                "image_id": candidate["id"],
                "main": score["main"],
                "aux": score["aux"],
                "scorer": args.model,
            }
        )
    return outputs


def parse_args(argv=None, api_key=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--caption-json",
        default=str(ROOT / "stage-1/目标图像描述/circo_test_target_captions.json"),
        help="CIRCO target-caption JSON, aligned with manifest query order.",
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="https://www.autodl.art/api/v1")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--batch-candidates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start-query", type=int, default=0)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args(argv)
    if api_key is not None:
        args.api_key = api_key
    elif not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY")
    return args


def run_pipeline(args):
    if not args.api_key:
        raise ValueError("Set API_KEY or pass --api-key")
    rows = load_jsonl(args.manifest)
    caption_rows = (
        json.load(open(args.caption_json, "r", encoding="utf-8"))
        if args.caption_json
        else [{} for _ in rows]
    )
    if len(caption_rows) != len(rows):
        raise ValueError(f"caption rows={len(caption_rows)}, manifest rows={len(rows)}")
    end = len(rows)
    if args.limit_queries is not None:
        end = min(end, args.start_query + args.limit_queries)
    selected_rows = list(enumerate(rows[args.start_query : end], args.start_query))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(output)
    print(f"queries selected: {len(selected_rows)}")
    print(f"existing candidate scores: {len(done)}")
    written = 0
    with open(output, "a", encoding="utf-8", buffering=1) as handle:
        for query_index, row in selected_rows:
            candidate_ids = [norm_name(value) for value in row["candidate_ids"]]
            candidate_paths = row.get("candidate_paths") or [None] * len(candidate_ids)
            query_id = str(row.get("query_id", f"toptee_{query_index}"))
            tasks = []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for start in range(0, len(candidate_ids), args.batch_candidates):
                    ids = candidate_ids[start : start + args.batch_candidates]
                    paths = candidate_paths[start : start + args.batch_candidates]
                    missing = [
                        offset
                        for offset, image_id in enumerate(ids)
                        if (query_id, image_id) not in done
                    ]
                    if not missing:
                        continue
                    tasks.append(
                        executor.submit(
                            score_batch,
                            args,
                            row,
                            caption_rows[query_index],
                            query_index,
                            start,
                            ids,
                            paths,
                        )
                    )
                for future in as_completed(tasks):
                    try:
                        result_rows = future.result()
                    except Exception as exc:
                        print(
                            f"WARNING query {query_index}: batch skipped: {exc}",
                            flush=True,
                        )
                        continue
                    for score_row in result_rows:
                        key = (
                            str(score_row["query_id"]),
                            norm_name(score_row["image_id"]),
                        )
                        if key in done:
                            continue
                        handle.write(json.dumps(score_row, ensure_ascii=False) + "\n")
                        done.add(key)
                        written += 1
            print(
                f"query {query_index + 1}/{end}; candidates={len(candidate_ids)}; new rows={written}"
            )
    print(f"saved: {output}")
    print(f"new candidate scores: {written}")


def main(argv=None, api_key=None):
    args = parse_args(argv, api_key)
    global requests
    import requests

    return run_pipeline(args)


def run(manifest, images, output, base_url, workers, api_key):
    return main(
        [
            "--manifest",
            str(manifest),
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
