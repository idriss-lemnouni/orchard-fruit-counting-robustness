"""Train a lightweight apple-center detector and audit held-out image counts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as f
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

WIDTH, HEIGHT, STRIDE = 288, 512, 4
SEED = 2041
MODEL_THRESHOLDS = (0.08, 0.12, 0.16, 0.20, 0.25, 0.32, 0.40, 0.55, 0.70, 0.85)
COLOR_SETTINGS = [(red, area) for red in (15, 25, 35, 50) for area in (4, 9, 16, 25)]


def resized_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR))


def mask_centers(path):
    with Image.open(path) as mask:
        rgb = np.asarray(mask.convert("RGB"), dtype=np.uint32)
    code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    values, inverse = np.unique(code.reshape(-1), return_inverse=True)
    y, x = np.indices(code.shape, dtype=np.int32)
    pixels = np.bincount(inverse)
    x = np.bincount(inverse, weights=x.ravel()) / pixels / code.shape[1]
    y = np.bincount(inverse, weights=y.ravel()) / pixels / code.shape[0]
    return np.stack((x[values != 0], y[values != 0]), axis=-1)


def center_targets(centers):
    rows, cols = HEIGHT // STRIDE, WIDTH // STRIDE
    target = np.zeros((rows, cols), dtype=np.float32)
    points = {(int(y * rows), int(x * cols)) for x, y in centers}
    for row, col in points:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = row + dy, col + dx
                if 0 <= y < rows and 0 <= x < cols:
                    target[y, x] = max(target[y, x], 0.6 ** (abs(dy) + abs(dx)))
    return target, len(centers) - len(points)


def encode_image(rgb):
    tensor = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32) / 255
    mean_rgb = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    sd_rgb = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean_rgb) / sd_rgb


class AppleCenters(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features[:4]
        self.features.requires_grad_(False)
        self.features.eval()
        self.head = nn.Sequential(
            nn.Conv2d(40, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 1, 1),
        )
        nn.init.constant_(self.head[-1].bias, -2.2)

    def forward(self, inputs):
        with torch.no_grad():
            detail = self.features[:2](inputs)
            context = self.features[2:](detail)
        upsampled = f.interpolate(context, size=detail.shape[-2:], mode="nearest")
        return self.head(torch.cat((detail, upsampled), dim=1)).squeeze(1)


def focal_loss(logits, target):
    probability = logits.sigmoid()
    positive = target == 1
    negative = ~positive
    loss = -f.logsigmoid(logits) * (1 - probability).square() * positive
    loss -= f.logsigmoid(-logits) * probability.square() * (1 - target).pow(4) * negative
    return loss.sum() / positive.sum().clamp(min=1)


def count_peaks(heatmap, threshold):
    heat = torch.as_tensor(heatmap)
    local_max = f.max_pool2d(heat[None, None], 3, stride=1, padding=1)[0, 0]
    peaks = ((heat >= threshold) & (heat == local_max)).numpy()
    # Equal neighboring scores form one plateau.
    return int(ndimage.label(peaks, np.ones((3, 3)))[1])


def count_color(rgb, red_margin, least_area):
    pixels = rgb.astype(np.int16)
    red_advantage = pixels[..., 0] - np.maximum(pixels[..., 1], pixels[..., 2])
    labels, _ = ndimage.label(red_advantage >= red_margin, np.ones((3, 3)))
    return int((np.bincount(labels.ravel())[1:] >= least_area).sum())


def has_overlap(boxes):
    edges = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(edges) < 2:
        return False
    x, y, w, h = edges.T
    width = (np.minimum.outer(x + w, x + w) - np.maximum.outer(x, x)).clip(0)
    height = (np.minimum.outer(y + h, y + h) - np.maximum.outer(y, y)).clip(0)
    intersection = width * height
    area = w * h
    union = area[:, None] + area[None, :] - intersection
    return bool(np.triu(intersection > 0.1 * union, 1).any())


def summarize(rows):
    result = {"images": len(rows), "apples": sum(row["truth"] for row in rows)}
    for method in ("learned", "color"):
        errors = [row[method] - row["truth"] for row in rows]
        if errors:
            result[method] = {"mae": mean(abs(error) for error in errors), "bias": mean(errors),
                              "rmse": mean(error ** 2 for error in errors) ** 0.5}
    return result


def train_and_evaluate(subset, out, epochs, batch_size, threads):
    if min(epochs, batch_size, threads) < 1:
        raise ValueError("Epochs, batch size and threads must be positive")
    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    spec = json.loads((subset / "subset.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=False)
    labeled_centers = spec.get("train_centers")

    def labels(name):
        if labeled_centers is None:
            return mask_centers(subset / "mask" / name)
        return labeled_centers[name]

    train_images, maps, collisions = [], [], 0
    for name in spec["train"]:
        train_images.append(encode_image(resized_rgb(subset / "train" / name)))
        target, missed = center_targets(labels(name))
        collisions += missed
        maps.append(torch.from_numpy(target))
    train_data = torch.stack(train_images)
    train_targets = torch.stack(maps)
    model = AppleCenters(pretrained=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.002)
    history = []
    for epoch in range(epochs):
        losses = []
        for indices in torch.randperm(len(train_data)).split(batch_size):
            optimizer.zero_grad()
            loss = focal_loss(model(train_data[indices]), train_targets[indices])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(mean(losses))
        print(f"Epoch {epoch + 1}/{epochs}, loss {history[-1]:.5f}", flush=True)

    calibration = []
    model.eval()
    with torch.inference_mode():
        for name in spec["calibration"]:
            rgb = resized_rgb(subset / "train" / name)
            heat = model(encode_image(rgb)[None])[0].sigmoid().numpy()
            n = len(labels(name))
            calibration.append((rgb, heat, n))
        threshold = min(MODEL_THRESHOLDS, key=lambda value:
                        mean(abs(count_peaks(heat, value) - n) for _, heat, n in calibration))
        red, min_area = min(COLOR_SETTINGS, key=lambda params:
                            mean(abs(count_color(rgb, *params) - n) for rgb, _, n in calibration))
        rows = []
        for record in spec["test"]:
            rgb = resized_rgb(subset / "test" / record["filename"])
            heat = model(encode_image(rgb)[None])[0].sigmoid().numpy()
            boxes = record["boxes"]
            areas = [box[2] * box[3] for box in boxes]
            rows.append({"group": record["group"], "truth": len(boxes),
                         "small_majority": bool(areas) and median(areas) < 32 ** 2,
                         "overlap": has_overlap(boxes), "learned": count_peaks(heat, threshold),
                         "color": count_color(rgb, red, min_area)})

    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    metrics = {
        "protocol": {"seed": SEED, "epochs": epochs, "batch_size": batch_size, "threads": threads,
                     "design": spec.get("design", "official_train_and_detection_test"),
                     "input_width": WIDTH, "input_height": HEIGHT, "train_images": len(spec["train"]),
                     "calibration_images": len(calibration), "target_stride": STRIDE,
                     "colliding_training_centers": collisions, "coco_sha256": spec["coco_sha256"],
                     "archive_md5_verified": spec["archive_md5_verified"],
                     "threshold": threshold, "color_red_margin": red, "color_min_area": min_area},
        "training_loss": history,
        "calibration": summarize([{"truth": n, "learned": count_peaks(heat, threshold),
                                  "color": count_color(rgb, red, min_area)} for rgb, heat, n in calibration]),
        "test": summarize(rows), "by_row_side": {name: summarize(group) for name, group in sorted(groups.items())},
        "by_size": {"median_below_1024_px2": summarize([row for row in rows if row["small_majority"]]),
                    "other": summarize([row for row in rows if not row["small_majority"]])},
        "by_overlap": {"any_box_iou_above_0_1": summarize([row for row in rows if row["overlap"]]),
                       "other": summarize([row for row in rows if not row["overlap"]])},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=Path("runs/subset"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    train_and_evaluate(args.subset, args.out, args.epochs, args.batch_size, args.threads)
