import argparse
import csv
import json
import math
import random
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for visualization. Install it with `pip install matplotlib`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fallback_retrieval_average_precision(preds, target, top_k=None):
    top_k = top_k or preds.shape[-1]
    top_indices = preds.topk(min(top_k, preds.shape[-1]), sorted=True, dim=-1)[1]
    target = target[top_indices]

    if not target.sum():
        return torch.tensor(0.0, device=preds.device)

    positions = torch.arange(
        1, len(target) + 1, device=target.device, dtype=torch.float32
    )[target > 0]
    return (
        (torch.arange(len(positions), device=positions.device, dtype=torch.float32) + 1)
        / positions
    ).mean()


def _install_import_stubs():
    try:
        import pytorch_lightning as _pl  # noqa: F401
    except ModuleNotFoundError:
        lightning_stub = types.ModuleType("pytorch_lightning")

        class _LightningModule(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.trainer = None
                self.global_step = 0

            def log(self, *args, **kwargs):
                return None

        lightning_stub.LightningModule = _LightningModule
        sys.modules["pytorch_lightning"] = lightning_stub

    try:
        from torchmetrics.functional import (  # noqa: F401
            retrieval_average_precision as _tm_retrieval_average_precision,
        )
    except ModuleNotFoundError:
        torchmetrics_stub = types.ModuleType("torchmetrics")
        torchmetrics_functional_stub = types.ModuleType("torchmetrics.functional")
        torchmetrics_functional_stub.retrieval_average_precision = (
            _fallback_retrieval_average_precision
        )
        torchmetrics_stub.functional = torchmetrics_functional_stub
        sys.modules["torchmetrics"] = torchmetrics_stub
        sys.modules["torchmetrics.functional"] = torchmetrics_functional_stub

    try:
        import ftfy as _ftfy  # noqa: F401
    except ModuleNotFoundError:
        ftfy_stub = types.ModuleType("ftfy")

        def _fix_text(text):
            return text

        ftfy_stub.fix_text = _fix_text
        sys.modules["ftfy"] = ftfy_stub


_install_import_stubs()

from src.model import ZS_SBIR
from src.sketchy_dataset import normal_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def canonicalize_class_name(name):
    return str(name).replace("_", " ").replace("-", " ").strip().lower()


def select_evenly(paths, limit):
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    return [paths[math.floor(index * len(paths) / limit)] for index in range(limit)]


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int
    classname: str
    domain: str


class ClassFolderDataset(Dataset):
    def __init__(self, root, split_name, classnames, max_size, domain, max_samples_per_class=0):
        self.root = Path(root)
        self.split_name = split_name
        self.classnames = list(classnames)
        self.max_size = max_size
        self.domain = domain
        self.max_samples_per_class = max_samples_per_class
        self.transform = normal_transform()
        self.records = []
        self.resolved_dirs = {}
        self.stats = {}
        self.class_to_paths = defaultdict(list)

        if not self.root.is_dir():
            raise FileNotFoundError(f"Missing {domain} root: {self.root}")

        available_dirs = {
            canonicalize_class_name(path.name): path.name
            for path in self.root.iterdir()
            if path.is_dir()
        }

        missing_classes = []
        for label, classname in enumerate(self.classnames):
            class_dir_name = self._resolve_dir_name(classname, available_dirs)
            if class_dir_name is None:
                missing_classes.append(classname)
                continue

            class_dir = self.root / class_dir_name
            all_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not all_paths:
                missing_classes.append(classname)
                continue

            selected_paths = select_evenly(all_paths, self.max_samples_per_class)
            self.resolved_dirs[classname] = class_dir_name
            self.stats[classname] = {
                "resolved_dir": class_dir_name,
                "available": len(all_paths),
                "used": len(selected_paths),
            }
            self.class_to_paths[classname].extend(selected_paths)
            self.records.extend(
                SampleRecord(
                    path=path,
                    label=label,
                    classname=classname,
                    domain=self.domain,
                )
                for path in selected_paths
            )

        if missing_classes:
            raise FileNotFoundError(
                f"Missing or empty class folders under {self.root}: {missing_classes}"
            )

        if not self.records:
            raise RuntimeError(f"No samples found under {self.root}")

    @staticmethod
    def _resolve_dir_name(classname, available_dirs):
        candidates = [
            classname,
            classname.replace("_", " "),
            classname.replace(" ", "_"),
            classname.replace("-", " "),
            classname.replace(" ", "-"),
        ]
        for candidate in candidates:
            if candidate in available_dirs.values():
                return candidate

        return available_dirs.get(canonicalize_class_name(classname))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = ImageOps.pad(
            Image.open(record.path).convert("RGB"),
            size=(self.max_size, self.max_size),
        )
        tensor = self.transform(image)
        return tensor, record.label, record.classname, str(record.path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize CoPrompt embeddings for selected classes from photo/<class> and sketch/<class> folders."
        )
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Class names to visualize, for example: cow raccoon scissors seagull sword tree",
    )
    parser.add_argument(
        "--prompt_classes",
        nargs="+",
        default=None,
        help=(
            "Optional class names used to build CoPrompt prompts. "
            "If omitted, the script uses the same list as --classes."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="Root folder that contains photo/ and sketch/, for example /kaggle/input/.../Sketchy",
    )
    parser.add_argument(
        "--photo_root",
        type=str,
        default="",
        help="Folder containing photo/<class>. Example: /kaggle/input/.../Sketchy/photo",
    )
    parser.add_argument(
        "--sketch_root",
        type=str,
        default="",
        help="Folder containing sketch/<class>. Example: /kaggle/input/.../Sketchy/sketch",
    )
    parser.add_argument("--photo_subdir", type=str, default="photo")
    parser.add_argument("--sketch_subdir", type=str, default="sketch")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save plots and metadata. Defaults to visualize/outputs/<ckpt_name>_<classes>",
    )
    parser.add_argument("--dataset", type=str, default="sketchy_2")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--img_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--distill", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use_classes", type=int, default=104)
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default="Co_prompt")
    parser.add_argument("--use_adapt_sk", type=str2bool, default=True)
    parser.add_argument("--use_adapt_ph", type=str2bool, default=True)
    parser.add_argument("--use_adapt_txt", type=str2bool, default=True)
    parser.add_argument("--use_co_sk", type=str2bool, default=True)
    parser.add_argument("--use_co_ph", type=str2bool, default=True)
    parser.add_argument("--progress", type=str2bool, default=False)
    parser.add_argument("--gzs", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=0,
        help="Limit the number of samples used per class and per domain. Use 0 for all samples.",
    )
    parser.add_argument(
        "--grid_samples_per_class",
        type=int,
        default=4,
        help="Number of thumbnails per class for the saved sample grid.",
    )
    parser.add_argument(
        "--save_combined_grid",
        type=str2bool,
        default=False,
        help="Whether to also save a combined photo+sketch grid.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    return parser.parse_args()


def normalize_split_root(root, split_name, classnames):
    path = Path(root)
    if not path.exists():
        return path

    if path.parent.name == split_name:
        known_classnames = {canonicalize_class_name(classname) for classname in classnames}
        if canonicalize_class_name(path.name) in known_classnames:
            return path.parent

    return path


def infer_sibling_root(known_root, from_split, to_split):
    known_root = Path(known_root)
    if known_root.name != from_split:
        raise ValueError(
            f"Cannot infer `{to_split}` root from `{known_root}`. "
            f"Expected the known root folder to end with `{from_split}`."
        )
    return known_root.parent / to_split


def resolve_split_roots(args):
    photo_root = Path(args.photo_root) if args.photo_root else None
    sketch_root = Path(args.sketch_root) if args.sketch_root else None

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        photo_root = photo_root or (dataset_root / args.photo_subdir)
        sketch_root = sketch_root or (dataset_root / args.sketch_subdir)

    if photo_root is None and sketch_root is None:
        raise ValueError(
            "Provide --dataset_root or at least one of --photo_root / --sketch_root."
        )

    if photo_root is not None:
        photo_root = normalize_split_root(photo_root, args.photo_subdir, args.classes)
    if sketch_root is not None:
        sketch_root = normalize_split_root(sketch_root, args.sketch_subdir, args.classes)

    if photo_root is None:
        sketch_root = normalize_split_root(sketch_root, args.sketch_subdir, args.classes)
        photo_root = infer_sibling_root(sketch_root, args.sketch_subdir, args.photo_subdir)
    if sketch_root is None:
        photo_root = normalize_split_root(photo_root, args.photo_subdir, args.classes)
        sketch_root = infer_sibling_root(photo_root, args.photo_subdir, args.sketch_subdir)

    return photo_root, sketch_root


def load_checkpoint_state(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = dict(state_dict)
    skip_keys = [
        "model.prompt_learner_photo.token_prefix",
        "model.prompt_learner_photo.token_suffix",
        "model.prompt_learner_sketch.token_prefix",
        "model.prompt_learner_sketch.token_suffix",
    ]
    for key in skip_keys:
        state_dict.pop(key, None)
    return state_dict


def build_model(args, prompt_classnames, device):
    model = ZS_SBIR(args=args, classname=prompt_classnames)
    state_dict = load_checkpoint_state(args.ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if device.type == "cpu":
        model.float()
    return model, list(missing), list(unexpected)


def extract_embeddings(model, dataloader, prompt_classnames, image_type, device):
    features = []
    metadata = []

    with torch.no_grad():
        for images, labels, classnames, paths in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            batch_features = model.model.extract_feature(
                images,
                classname=prompt_classnames,
                type=image_type,
            )
            batch_features = batch_features.detach().cpu().float()
            labels = labels.detach().cpu().tolist()

            features.append(batch_features)
            metadata.extend(
                {
                    "label": int(label),
                    "class_name": classname,
                    "domain": image_type,
                    "path": path,
                }
                for label, classname, path in zip(labels, classnames, paths)
            )

    return torch.cat(features, dim=0), metadata


def compute_pca_projection(features):
    features = features.float()
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Need at least two samples to compute a 2D projection.")

    centered = features - features.mean(dim=0, keepdim=True)
    if torch.allclose(centered.abs().sum(), torch.tensor(0.0)):
        coords = torch.zeros((features.shape[0], 2), dtype=torch.float32)
        return coords, [0.0, 0.0]

    q = min(8, centered.shape[0], centered.shape[1])
    q = max(2, q)
    _, singular_values, right_vectors = torch.pca_lowrank(centered, q=q)
    coords = centered @ right_vectors[:, :2]

    if coords.shape[1] == 1:
        coords = torch.cat(
            [coords, torch.zeros((coords.shape[0], 1), dtype=coords.dtype)], dim=1
        )

    total_variance = centered.var(dim=0, unbiased=False).sum()
    if float(total_variance) > 0:
        explained = (singular_values[:2] ** 2) / max(centered.shape[0] - 1, 1)
        explained_ratio = (explained / total_variance).tolist()
    else:
        explained_ratio = [0.0, 0.0]

    if len(explained_ratio) < 2:
        explained_ratio.append(0.0)

    return coords[:, :2], explained_ratio[:2]


def build_color_map(classnames):
    cmap_name = "tab10" if len(classnames) <= 10 else "tab20"
    cmap = plt.cm.get_cmap(cmap_name, len(classnames))
    return {classname: cmap(index) for index, classname in enumerate(classnames)}


def save_embedding_plot(points, classnames, explained_ratio, ckpt_name, output_path):
    colors = build_color_map(classnames)
    markers = {"photo": "o", "sketch": "^"}

    fig, ax = plt.subplots(figsize=(13, 9), dpi=220)

    for domain in ["photo", "sketch"]:
        for classname in classnames:
            subset = [
                point
                for point in points
                if point["domain"] == domain and point["class_name"] == classname
            ]
            if not subset:
                continue

            ax.scatter(
                [point["x"] for point in subset],
                [point["y"] for point in subset],
                s=38,
                alpha=0.8,
                c=[colors[classname]],
                marker=markers[domain],
                linewidths=0.2,
                edgecolors="black",
            )

    for classname in classnames:
        class_points = [point for point in points if point["class_name"] == classname]
        if not class_points:
            continue

        class_center_x = sum(point["x"] for point in class_points) / len(class_points)
        class_center_y = sum(point["y"] for point in class_points) / len(class_points)
        ax.text(
            class_center_x,
            class_center_y,
            classname,
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )

        photo_points = [
            point for point in class_points if point["domain"] == "photo"
        ]
        sketch_points = [
            point for point in class_points if point["domain"] == "sketch"
        ]
        if photo_points and sketch_points:
            photo_center = (
                sum(point["x"] for point in photo_points) / len(photo_points),
                sum(point["y"] for point in photo_points) / len(photo_points),
            )
            sketch_center = (
                sum(point["x"] for point in sketch_points) / len(sketch_points),
                sum(point["y"] for point in sketch_points) / len(sketch_points),
            )
            ax.plot(
                [photo_center[0], sketch_center[0]],
                [photo_center[1], sketch_center[1]],
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                color=colors[classname],
            )

    ax.set_title(f"CoPrompt Feature Projection: {ckpt_name}", fontsize=15, weight="bold")
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.2f}% variance)")
    ax.grid(alpha=0.2, linestyle="--")

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=classname,
            markerfacecolor=colors[classname],
            markeredgecolor="black",
            markersize=8,
        )
        for classname in classnames
    ]
    domain_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[domain],
            color="black",
            linestyle="None",
            label=domain,
            markersize=8,
        )
        for domain in ["photo", "sketch"]
    ]

    class_legend = ax.legend(
        handles=class_handles,
        title="Class",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    ax.add_artist(class_legend)
    ax.legend(
        handles=domain_handles,
        title="Domain",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.48),
    )

    fig.tight_layout(rect=(0, 0, 0.82, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_thumbnail(image_path, size):
    image = Image.open(image_path).convert("RGB")
    return ImageOps.pad(image, (size, size), color=(255, 255, 255))


def save_single_domain_grid(dataset, classnames, samples_per_class, title, output_path):
    thumb_size = 112
    padding = 12
    label_width = 170
    header_height = 52
    row_height = thumb_size + padding
    grid_width = max(samples_per_class, 1) * (thumb_size + padding)

    width = padding * 3 + label_width + grid_width
    height = padding * 2 + header_height + len(classnames) * row_height

    canvas = Image.new("RGB", (width, height), color=(248, 247, 243))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x_class = padding
    x_samples = padding * 2 + label_width

    draw.text((x_class, padding), "Class", fill=(20, 20, 20), font=font)
    draw.text((x_samples, padding), title, fill=(20, 20, 20), font=font)

    for row_index, classname in enumerate(classnames):
        y = padding + header_height + row_index * row_height
        draw.text((x_class, y + thumb_size // 2 - 6), classname, fill=(20, 20, 20), font=font)

        image_paths = dataset.class_to_paths[classname][:samples_per_class]
        for col_index, image_path in enumerate(image_paths):
            thumb = load_thumbnail(image_path, thumb_size)
            canvas.paste(
                thumb,
                (x_samples + col_index * (thumb_size + padding), y),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_sample_grid(photo_dataset, sketch_dataset, classnames, samples_per_class, output_path):
    thumb_size = 96
    padding = 12
    label_width = 150
    photo_title_width = max(samples_per_class, 1) * (thumb_size + padding)
    sketch_title_width = max(samples_per_class, 1) * (thumb_size + padding)
    header_height = 54
    row_height = thumb_size + padding

    width = (
        padding * 3
        + label_width
        + photo_title_width
        + sketch_title_width
    )
    height = padding * 2 + header_height + len(classnames) * row_height

    canvas = Image.new("RGB", (width, height), color=(248, 247, 243))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x_class = padding
    x_photo = padding * 2 + label_width
    x_sketch = x_photo + photo_title_width

    draw.text((x_class, padding), "Class", fill=(20, 20, 20), font=font)
    draw.text((x_photo, padding), "Photo samples", fill=(20, 20, 20), font=font)
    draw.text((x_sketch, padding), "Sketch samples", fill=(20, 20, 20), font=font)

    for row_index, classname in enumerate(classnames):
        y = padding + header_height + row_index * row_height
        draw.text((x_class, y + thumb_size // 2 - 6), classname, fill=(20, 20, 20), font=font)

        photo_paths = photo_dataset.class_to_paths[classname][:samples_per_class]
        sketch_paths = sketch_dataset.class_to_paths[classname][:samples_per_class]

        for col_index, image_path in enumerate(photo_paths):
            thumb = load_thumbnail(image_path, thumb_size)
            canvas.paste(
                thumb,
                (x_photo + col_index * (thumb_size + padding), y),
            )

        for col_index, image_path in enumerate(sketch_paths):
            thumb = load_thumbnail(image_path, thumb_size)
            canvas.paste(
                thumb,
                (x_sketch + col_index * (thumb_size + padding), y),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_points_csv(points, output_path):
    fieldnames = ["class_name", "domain", "label", "x", "y", "path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "class_name": point["class_name"],
                    "domain": point["domain"],
                    "label": point["label"],
                    "x": f"{point['x']:.8f}",
                    "y": f"{point['y']:.8f}",
                    "path": point["path"],
                }
            )


def make_output_dir(args):
    if args.output_dir:
        return Path(args.output_dir)

    class_slug = "_".join(
        canonicalize_class_name(classname).replace(" ", "-") for classname in args.classes
    )
    ckpt_stem = Path(args.ckpt_path).stem
    return REPO_ROOT / "visualize" / "outputs" / f"{ckpt_stem}_{class_slug}"


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    prompt_classnames = list(args.prompt_classes or args.classes)
    photo_root, sketch_root = resolve_split_roots(args)

    photo_dataset = ClassFolderDataset(
        root=photo_root,
        split_name=args.photo_subdir,
        classnames=args.classes,
        max_size=args.max_size,
        domain="photo",
        max_samples_per_class=args.max_samples_per_class,
    )
    sketch_dataset = ClassFolderDataset(
        root=sketch_root,
        split_name=args.sketch_subdir,
        classnames=args.classes,
        max_size=args.max_size,
        domain="sketch",
        max_samples_per_class=args.max_samples_per_class,
    )

    dataloader_kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": args.workers,
        "shuffle": False,
        "pin_memory": args.device.startswith("cuda"),
    }
    photo_loader = DataLoader(photo_dataset, **dataloader_kwargs)
    sketch_loader = DataLoader(sketch_dataset, **dataloader_kwargs)

    device = torch.device(args.device)
    model, missing_keys, unexpected_keys = build_model(args, prompt_classnames, device)

    sketch_features, sketch_metadata = extract_embeddings(
        model=model,
        dataloader=sketch_loader,
        prompt_classnames=prompt_classnames,
        image_type="sketch",
        device=device,
    )
    photo_features, photo_metadata = extract_embeddings(
        model=model,
        dataloader=photo_loader,
        prompt_classnames=prompt_classnames,
        image_type="photo",
        device=device,
    )

    all_features = torch.cat([photo_features, sketch_features], dim=0)
    all_metadata = photo_metadata + sketch_metadata
    coords, explained_ratio = compute_pca_projection(all_features)

    points = []
    for meta, coord in zip(all_metadata, coords.tolist()):
        point = dict(meta)
        point["x"] = float(coord[0])
        point["y"] = float(coord[1])
        points.append(point)

    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_embedding_plot(
        points=points,
        classnames=args.classes,
        explained_ratio=explained_ratio,
        ckpt_name=ckpt_path.name,
        output_path=output_dir / "embedding_pca.png",
    )
    save_single_domain_grid(
        dataset=photo_dataset,
        classnames=args.classes,
        samples_per_class=max(args.grid_samples_per_class, 1),
        title="Photo samples",
        output_path=output_dir / "photo_grid.png",
    )
    save_single_domain_grid(
        dataset=sketch_dataset,
        classnames=args.classes,
        samples_per_class=max(args.grid_samples_per_class, 1),
        title="Sketch samples",
        output_path=output_dir / "sketch_grid.png",
    )
    if args.save_combined_grid:
        save_sample_grid(
            photo_dataset=photo_dataset,
            sketch_dataset=sketch_dataset,
            classnames=args.classes,
            samples_per_class=max(args.grid_samples_per_class, 1),
            output_path=output_dir / "sample_grid.png",
        )
    save_points_csv(points, output_dir / "points.csv")

    summary = {
        "checkpoint": str(ckpt_path),
        "photo_root": str(photo_root),
        "sketch_root": str(sketch_root),
        "classes": list(args.classes),
        "prompt_classes": prompt_classnames,
        "num_photo_samples": len(photo_dataset),
        "num_sketch_samples": len(sketch_dataset),
        "explained_variance_ratio": {
            "pc1": explained_ratio[0],
            "pc2": explained_ratio[1],
        },
        "photo_stats": photo_dataset.stats,
        "sketch_stats": sketch_dataset.stats,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "outputs": {
            "embedding_plot": str(output_dir / "embedding_pca.png"),
            "photo_grid": str(output_dir / "photo_grid.png"),
            "sketch_grid": str(output_dir / "sketch_grid.png"),
            "points_csv": str(output_dir / "points.csv"),
        },
    }
    if args.save_combined_grid:
        summary["outputs"]["sample_grid"] = str(output_dir / "sample_grid.png")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Checkpoint      : {ckpt_path}")
    print(f"Photo root      : {photo_root}")
    print(f"Sketch root     : {sketch_root}")
    print(f"Classes         : {args.classes}")
    print(f"Prompt classes  : {prompt_classnames}")
    print(f"Photo samples   : {len(photo_dataset)}")
    print(f"Sketch samples  : {len(sketch_dataset)}")
    print(f"Saved plot      : {output_dir / 'embedding_pca.png'}")
    print(f"Saved photos    : {output_dir / 'photo_grid.png'}")
    print(f"Saved sketches  : {output_dir / 'sketch_grid.png'}")
    if args.save_combined_grid:
        print(f"Saved grid      : {output_dir / 'sample_grid.png'}")
    print(f"Saved CSV       : {output_dir / 'points.csv'}")
    print(f"Saved summary   : {output_dir / 'summary.json'}")
    if missing_keys:
        print(f"Missing keys    : {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys : {unexpected_keys}")


if __name__ == "__main__":
    main()
