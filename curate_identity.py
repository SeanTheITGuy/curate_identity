#!/usr/bin/env python3

"""
curate_identity.py

Identity-based image curation tool for synthetic LoRA datasets.

This script:

1. Builds a facial identity centroid from reference images
2. Scores candidate images against that centroid
3. Either:
    - sorts images into approved/review/reject
    - OR performs a dry-run ranking report

Useful for:
- iterative LoRA bootstrapping
- synthetic dataset cleanup
- identity drift detection
- filtering bad generations

------------------------------------------------------------
INSTALL
------------------------------------------------------------

pip install insightface onnxruntime pillow numpy tqdm scikit-learn

------------------------------------------------------------
USAGE
------------------------------------------------------------

Dry run only:

python curate_identity.py \
  --refs ./refs \
  --input ./gens \
  --output ./sorted \
  --dry-run

Actual sorting:

python curate_identity.py \
  --refs ./refs \
  --input ./gens \
  --output ./sorted

Top N report:

python curate_identity.py \
  --refs ./refs \
  --input ./gens \
  --output ./sorted \
  --dry-run \
  --top 100

------------------------------------------------------------
FOLDER STRUCTURE
------------------------------------------------------------

project/
├── refs/
├── gens/
├── sorted/
│   ├── approved/
│   ├── review/
│   └── reject/
└── curate_identity.py

------------------------------------------------------------
THRESHOLDS
------------------------------------------------------------

Typical values:

0.85+  extremely strong identity match
0.82+  excellent
0.75+  usable / review
0.65+  drift beginning
<0.65  likely different person

------------------------------------------------------------
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="`estimate` is deprecated since version 0.26"
)

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import insightface
from insightface.app import FaceAnalysis


SUPPORTED_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def normalize(v):
    return v / np.linalg.norm(v)


def similarity(a, b):
    return cosine_similarity(
        a.reshape(1, -1),
        b.reshape(1, -1)
    )[0][0]


def gather_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ]


def ensure_dirs(base):

    approved = base / "approved"
    review = base / "review"
    reject = base / "reject"

    approved.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    reject.mkdir(parents=True, exist_ok=True)

    return approved, review, reject


def copy_with_score(src, dst_dir, score):

    new_name = f"{score:.4f}_{src.name}"
    shutil.copy2(src, dst_dir / new_name)


def get_face_embedding(app, image_np):
    """
    Returns the embedding for the largest detected face.
    """

    faces = app.get(image_np)

    if not faces:
        return None

    # Choose largest face
    faces = sorted(
        faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        ),
        reverse=True
    )

    return faces[0].embedding


def build_centroid(app, ref_paths):

    embeddings = []

    print(f"\nBuilding identity centroid from {len(ref_paths)} images...\n")

    for path in tqdm(ref_paths):

        try:

            img = load_image(path)
            emb = get_face_embedding(app, img)

            if emb is None:
                print(f"[WARN] No face found in reference: {path}")
                continue

            embeddings.append(normalize(emb))

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    if not embeddings:
        raise RuntimeError("No usable reference embeddings found.")

    centroid = np.mean(embeddings, axis=0)
    centroid = normalize(centroid)

    print("\nCentroid created.\n")

    return centroid


def main():

    parser = argparse.ArgumentParser(
        description="Identity-based image curation tool"
    )

    parser.add_argument(
        "--refs",
        required=True,
        help="Reference image folder"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Candidate image folder"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder"
    )

    parser.add_argument(
        "--approved-threshold",
        type=float,
        default=0.82,
        help="Similarity threshold for approved images"
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for review images"
    )

    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="InsightFace detector size"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print ranked results without moving files"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only show top N results in dry-run mode"
    )

    args = parser.parse_args()

    ref_dir = Path(args.refs)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    approved_dir, review_dir, reject_dir = ensure_dirs(output_dir)

    print("\nLoading InsightFace model...\n")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(args.det_size, args.det_size)
    )

    ref_paths = gather_images(ref_dir)
    input_paths = gather_images(input_dir)

    print(f"Found {len(ref_paths)} reference images")
    print(f"Found {len(input_paths)} candidate images")

    centroid = build_centroid(app, ref_paths)

    stats = {
        "approved": 0,
        "review": 0,
        "reject": 0,
        "noface": 0
    }

    results = []

    print("\nScoring candidate images...\n")

    for path in tqdm(input_paths):

        try:

            img = load_image(path)
            emb = get_face_embedding(app, img)

            if emb is None:
                stats["noface"] += 1
                continue

            emb = normalize(emb)

            score = similarity(centroid, emb)

            results.append((score, path))

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    # --------------------------------------------------------
    # DRY RUN MODE
    # --------------------------------------------------------

    if args.dry_run:

        print("\n=== DRY RUN RESULTS ===\n")

        display_results = results

        if args.top > 0:
            display_results = results[:args.top]

        for score, path in display_results:

            if score >= args.approved_threshold:
                label = "APPROVED"

            elif score >= args.review_threshold:
                label = "REVIEW"

            else:
                label = "REJECT"

            print(f"{score:.4f} [{label}] {path}")

        print("\nDry run complete.\n")

        return

    # --------------------------------------------------------
    # FILE SORTING MODE
    # --------------------------------------------------------

    print("\nSorting files...\n")

    for score, path in tqdm(results):

        try:

            if score >= args.approved_threshold:

                copy_with_score(path, approved_dir, score)
                stats["approved"] += 1

            elif score >= args.review_threshold:

                copy_with_score(path, review_dir, score)
                stats["review"] += 1

            else:

                copy_with_score(path, reject_dir, score)
                stats["reject"] += 1

        except Exception as e:
            print(f"[ERROR] Copying {path}: {e}")

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\nDone.\n")

    print("Results:")
    print(f"  Approved : {stats['approved']}")
    print(f"  Review   : {stats['review']}")
    print(f"  Reject   : {stats['reject']}")
    print(f"  No Face  : {stats['noface']}")

    print("\nThresholds:")
    print(f"  Approved >= {args.approved_threshold}")
    print(f"  Review   >= {args.review_threshold}")

    print("\nSuggested next step:")
    print("Inspect review/ manually before training.\n")


if __name__ == "__main__":
    main()
