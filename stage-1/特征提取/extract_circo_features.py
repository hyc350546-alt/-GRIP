"""Portable CIRCO ViT-L extraction using the bundled IP-CIR CLIP implementation.

This is a path adapter, not a claim of bitwise equality across CUDA versions.
The default preprocessing and batch size are from the archived IP-CIR source.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.circo_workflow import external_workdir, load_queries, read_json, require_new


def extract(queries, coco_info, image_root, output, device="cuda"):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from shared.CLIP编码器.backbones import load_backbone

    output = Path(output)
    cache = output / "debiased"
    files = [output / "circo_test_L.npz", output / "source_features_L.npz"]
    files += [
        cache / f"{prefix}_predicted_features_{i}.pt"
        for prefix in ("gpt", "blip")
        for i in range(15)
    ]
    require_new(files)
    info = read_json(coco_info)["images"]
    ids = [str(row["id"]) for row in info]
    paths = [Path(image_root) / row["file_name"] for row in info]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gallery IDs in COCO image-info file")
    image_by_id = dict(zip(ids, paths))
    reference_paths = [image_by_id[row["reference_img_id"]] for row in queries]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} gallery images; first={missing[0]}"
        )
    # retrieval_circo.py explicitly converts CLIP to float32, even on CUDA.
    encoder = load_backbone("L", device=device, precision="fp32")
    cache.mkdir(parents=True, exist_ok=True)

    def encode_images(image_paths):
        features = []
        for start in range(0, len(image_paths), 32):
            batch = []
            for filename in image_paths[start : start + 32]:
                with Image.open(filename) as image:
                    batch.append(encoder.preprocess(image))
            with torch.no_grad():
                features.append(
                    encoder.model.encode_image(
                        torch.stack(batch).to(encoder.device)
                    ).cpu()
                )
            if start % 3200 == 0:
                print(
                    f"Images {min(start + 32, len(image_paths))}/{len(image_paths)}",
                    flush=True,
                )
        return torch.cat(features)

    gallery = encode_images(paths)
    torch.save({"feature": gallery, "name": ids}, files[0])
    references = encode_images(reference_paths)
    torch.save(
        {
            row["reference_img_id"]: references[i : i + 1]
            for i, row in enumerate(queries)
        },
        files[1],
    )
    for prefix, key in (("gpt", "target_captions"), ("blip", "before_captions")):
        for caption_index in range(15):
            chunks = []
            for start in range(0, len(queries), 32):
                captions = [
                    row[key][caption_index] for row in queries[start : start + 32]
                ]
                tokens = encoder.tokenizer(captions, truncate=True).to(encoder.device)
                with torch.no_grad():
                    # Normalize each caption before the 15-caption aggregation.
                    features = F.normalize(encoder.model.encode_text(tokens), dim=-1)
                chunks.append(features.cpu())
            torch.save(
                torch.cat(chunks),
                cache / f"{prefix}_predicted_features_{caption_index}.pt",
            )
            print(f"{prefix} captions {caption_index + 1}/15", flush=True)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-info", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    return extract(
        load_queries(),
        args.coco_info,
        args.image_root,
        external_workdir(args.output),
        args.device,
    )


if __name__ == "__main__":
    main()
