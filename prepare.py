"""Prepare a deterministic MinneApple train, calibration and official test subset."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory
import zipfile

DETECTION_BYTES = 1825397590
DETECTION_MD5 = "d24a2f70144f1a3c5a52ffef66c50630"
TEST_BYTES = 1251569589
TEST_MD5 = "0c24727552bc3e94cad6d8f1c851c887"
COCO_MEMBER = "test_data/detection/ground_truth.json"


def file_hash(path, algorithm):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def check_archive(path, size, md5):
    if path.stat().st_size != size or file_hash(path, "md5") != md5:
        raise ValueError(f"Archive differs from the university checksum: {path.name}")


def key(name):
    return hashlib.sha256(("minneapple-subset-v1:" + name).encode()).digest()


def source_kind(member):
    path = PurePosixPath(member.name)
    parts = tuple(piece.lower() for piece in path.parts)
    if not member.isfile() or path.suffix.lower() != ".png" or path.name.startswith("."):
        return None
    if "train" in parts and "masks" in parts:
        return "mask"
    if "train" in parts and "images" in parts:
        return "train"
    if "test" in parts:
        return "test"
    return None


def official_test(coco, limit):
    categories = {item["id"] for item in coco["categories"] if item["name"].lower() == "apple"}
    if not categories:
        raise ValueError("The official annotation file has no apple category")
    by_image = defaultdict(list)
    for item in coco["annotations"]:
        if item["category_id"] in categories and not item["iscrowd"]:
            by_image[item["image_id"]].append(item["bbox"])
    groups = defaultdict(list)
    for item in coco["images"]:
        name = item["filename"]
        match = re.fullmatch(r"(dataset\d+_(?:front|back))_\d+\.png", name, re.I)
        if not match:
            raise ValueError(f"Unexpected official test image name: {name}")
        groups[match[1]].append({"filename": name, "group": match[1], "width": item["width"],
                                "height": item["height"], "boxes": by_image[item["id"]]})
    return [record for group in sorted(groups)
            for record in sorted(groups[group], key=lambda row: key(row["filename"]))[:limit]]


def train_and_calibration_names(names, training, calibration):
    by_sequence = defaultdict(list)
    for name in names:
        match = re.fullmatch(r"(\d{8}_\d{6})_image\d+\.png", name, re.I)
        if not match:
            raise ValueError(f"Unexpected training frame name: {name}")
        by_sequence[match[1]].append(name)
    sequence_names = sorted(by_sequence, key=key)
    reserved = []
    while sequence_names and len(reserved) < calibration:
        reserved.extend(by_sequence[sequence_names.pop(0)])
    pool = [name for sequence in sequence_names for name in by_sequence[sequence]]
    if len(pool) < training or len(reserved) < calibration:
        raise ValueError("Available sequences cannot satisfy the requested disjoint subset")
    return set(sorted(pool, key=key)[:training]), set(sorted(reserved, key=key)[:calibration])


def prepare(detection, annotation, out, training, calibration, by_group, require_archive_md5=True):
    if min(training, calibration, by_group) < 1 or out.exists():
        raise ValueError("Use positive subset sizes and a fresh output directory")
    if require_archive_md5:
        check_archive(detection, DETECTION_BYTES, DETECTION_MD5)
    if annotation.suffix == ".zip":
        check_archive(annotation, TEST_BYTES, TEST_MD5)
        with zipfile.ZipFile(annotation) as archive:
            annotation_bytes = archive.read(COCO_MEMBER)
    else:
        annotation_bytes = annotation.read_bytes()
    test = official_test(json.loads(annotation_bytes), by_group)

    entries = defaultdict(set)
    with tarfile.open(detection, "r|gz") as archive:
        for item in archive:
            if kind := source_kind(item):
                name = PurePosixPath(item.name).name
                if name in entries[kind]:
                    raise ValueError(f"Duplicate {kind} image: {name}")
                entries[kind].add(name)
    common = sorted(entries["train"] & entries["mask"], key=key)
    test_names = {record["filename"] for record in test}
    if len(common) < training + calibration or not test_names <= entries["test"] or set(common) & test_names:
        raise ValueError(f"Dataset structure differs: {len(common)} pairs, {len(test_names - entries['test'])} missing official test frames")
    train_names, calibration_names = train_and_calibration_names(common, training, calibration)
    wanted = {"train": train_names | calibration_names, "mask": train_names | calibration_names, "test": test_names}
    result = {
        "archive_md5_verified": require_archive_md5,
        "coco_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "train": sorted(train_names), "calibration": sorted(calibration_names), "test": test,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temporary:
        stage = Path(temporary) / "subset"
        for kind in wanted:
            (stage / kind).mkdir(parents=True)
        with tarfile.open(detection, "r|gz") as archive:
            for item in archive:
                kind = source_kind(item)
                name = PurePosixPath(item.name).name
                if kind in wanted and name in wanted[kind]:
                    with archive.extractfile(item) as source, (stage / kind / name).open("wb") as stream:
                        shutil.copyfileobj(source, stream)
                    wanted[kind].remove(name)
        if any(wanted.values()):
            raise ValueError("Not all selected frames were extracted")
        (stage / "subset.json").write_text(json.dumps(result), encoding="utf-8")
        stage.rename(out)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection", type=Path, default=Path("data/raw/detection.tar.gz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/ground_truth.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/subset"))
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--calibration", type=int, default=28)
    parser.add_argument("--test-per-group", type=int, default=12)
    args = parser.parse_args()
    proof = prepare(args.detection, args.annotations, args.out, args.train, args.calibration, args.test_per_group)
    print({split: len(proof[split]) for split in ("train", "calibration", "test")})
