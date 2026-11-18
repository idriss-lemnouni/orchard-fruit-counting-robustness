"""Prepare a disjoint-row experiment from the official labeled test release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from zlib import crc32

from fetch_test_labels import ArchiveRangeReader, write_file
from prepare import COCO_MEMBER, official_test


def centers(row):
    return [[(x + width / 2) / row["width"], (y + height / 2) / row["height"]]
            for x, y, width, height in row["boxes"]]


def assign_rows(coco):
    rows = official_test(coco, 24)
    selected = {"train": [], "calibration": [], "test": []}
    limits = {1: 15, 2: 6, 3: 8, 4: 8}
    per_group = {}
    for row in rows:
        group = row["group"]
        number = int(group.split("_")[0].removeprefix("dataset"))
        if number not in limits:
            raise ValueError(f"Unexpected orchard row: {number}")
        per_group[group] = per_group.get(group, 0) + 1
        if per_group[group] <= limits[number]:
            split = {1: "train", 2: "calibration"}.get(number, "test")
            selected[split].append(row)
    if [len(selected[split]) for split in selected] != [30, 12, 24]:
        raise ValueError("Official row groups differ from the expected benchmark")
    return selected


def prepare(out, workers):
    with ZipFile(ArchiveRangeReader()) as archive:
        annotation = archive.read(COCO_MEMBER)
    selected = assign_rows(json.loads(annotation))
    images = []
    for split, records in selected.items():
        folder = out / ("test" if split == "test" else "train")
        folder.mkdir(parents=True, exist_ok=True)
        images.extend((record, folder) for record in records)

    def batch(entries):
        with ZipFile(ArchiveRangeReader()) as archive:
            for record, folder in entries:
                name = record["filename"]
                selected_image = archive.getinfo("test_data/detection/images/" + name)
                target = folder / name
                if target.exists() and crc32(target.read_bytes()) == selected_image.CRC:
                    continue
                write_file(target, archive.read(selected_image))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(batch, images[index::workers]) for index in range(workers)]
        for job in jobs:
            job.result()
    spec = {
        "archive_md5_verified": False,
        "coco_sha256": hashlib.sha256(annotation).hexdigest(),
        "design": "within_official_test_release_by_orchard_row",
        "train": [row["filename"] for row in selected["train"]],
        "calibration": [row["filename"] for row in selected["calibration"]],
        "train_centers": {row["filename"]: centers(row) for row in selected["train"] + selected["calibration"]},
        "test": selected["test"],
    }
    if any(not (0 <= x < 1 and 0 <= y < 1)
           for values in spec["train_centers"].values() for x, y in values):
        raise ValueError("An annotated center is outside the source image")
    write_file(out / "subset.json", json.dumps(spec).encode())
    print({split: len(selected[split]) for split in selected})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=Path("runs/row-holdout"))
    cli.add_argument("--workers", type=int, default=6)
    args = cli.parse_args()
    if args.workers < 1:
        cli.error("Positive worker count required")
    prepare(args.out, args.workers)
