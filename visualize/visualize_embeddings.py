import argparse
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _fallback_retrieval_average_precision(preds, target, top_k=None):
    top_k = top_k or preds.shape[-1]
    top_indices = preds.topk(min(top_k, preds.shape[-1]), sorted=True, dim=-1)[1]
    target = target[top_indices]
    if not target.sum():
        return torch.tensor(0.0, device=preds.device)

    positions = torch.arange(
        1, len(target) + 1, device=target.device, dtype=torch.float32
    )[target > 0]
    numerators = torch.arange(
        len(positions), device=positions.device, dtype=torch.float32
    ) + 1
    return (numerators / positions).mean()


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
    from torchmetrics.functional import retrieval_average_precision as _tm_retrieval_average_precision  # noqa: F401
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

from src.model import ZS_SBIR
from src.sketchy_dataset import normal_transform


def canonicalize_class_name(name):
    return name.replace("_", " ").replace("-", " ").strip().lower()


def sample_paths(paths, limit):
    if limit <= 0 or len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    return [paths[index] for index in indices]


class FolderDataset(Dataset):
    def __init__(self, root, classnames, max_size, max_samples_per_class):
        self.root = Path(root)
        self.classnames = list(classnames)
        self.max_size = max_size
        self.transform = normal_transform()
        self.samples = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Missing directory: {self.root}")

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
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            paths = sample_paths(paths, max_samples_per_class)
            if not paths:
                missing_classes.append(classname)
                continue

            self.samples.extend((path, label, classname) for path in paths)

        if missing_classes:
            raise FileNotFoundError(
                f"Missing or empty class folders under {self.root}: {missing_classes}"
            )

        if not self.samples:
            raise RuntimeError(f"No image samples found under {self.root}")

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
        return len(self.samples)

    def __getitem__(self, index):
        path, label, classname = self.samples[index]
        image = ImageOps.pad(
            Image.open(path).convert("RGB"),
            size=(self.max_size, self.max_size),
        )
        return self.transform(image), label, classname


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize photo and sketch embeddings with t-SNE."
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--photo_subdir", type=str, default="photo")
    parser.add_argument("--sketch_subdir", type=str, default="sketch")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=None)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_samples_per_class", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def infer_model_hparams(state_dict, args):
    if args.n_ctx is None:
        ctx = state_dict.get("model.prompt_learner_photo.ctx")
        args.n_ctx = int(ctx.shape[0]) if ctx is not None else 2

    if args.prompt_depth is None:
        pattern = re.compile(
            r"model\.prompt_learner_photo\.compound_prompt_projections\.(\d+)\.weight"
        )
        indices = []
        for key in state_dict:
            match = pattern.fullmatch(key)
            if match:
                indices.append(int(match.group(1)))
        args.prompt_depth = (max(indices) + 2) if indices else 12

    return args


def build_model(args, classnames, device):
    state_dict = load_checkpoint_state(args.ckpt_path)
    args = infer_model_hparams(state_dict, args)
    model = ZS_SBIR(args=args, classname=classnames)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if device.type == "cpu":
        model.float()
    return model, missing, unexpected


def extract_embeddings(model, dataloader, classnames, image_type, device):
    features = []
    sample_classnames = []

    with torch.no_grad():
        for images, _, batch_classnames in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            batch_features = model.model.extract_feature(
                images,
                classname=classnames,
                type=image_type,
            )
            features.append(F.normalize(batch_features.float(), dim=1).cpu())
            sample_classnames.extend(batch_classnames)

    return torch.cat(features, dim=0), sample_classnames


def run_tsne(features, perplexity, seed):
    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for t-SNE. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    total_samples = len(features)
    if total_samples < 2:
        raise ValueError("t-SNE needs at least 2 samples.")

    effective_perplexity = min(perplexity, total_samples - 1)
    if effective_perplexity < 1:
        raise ValueError(
            f"Perplexity must be >= 1 after adjustment, got {effective_perplexity}."
        )

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=effective_perplexity,
    )
    return tsne.fit_transform(features.numpy()), effective_perplexity


def plot_domain(ax, coords, classnames, all_classnames, colors, title):
    classnames = np.array(classnames)
    for classname in all_classnames:
        mask = classnames == classname
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=20,
                alpha=0.9,
                c=[colors[classname]],
                edgecolors="white",
                linewidths=0.35,
                label=classname,
            )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def resolve_output_path(args):
    if args.output_path:
        return Path(args.output_path)
    return (
        REPO_ROOT
        / "visualize"
        / "outputs"
        / f"{Path(args.ckpt_path).stem}_tsne.png"
    )


def main():
    args = parse_args()
    seed_everything(args.seed)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_dir = Path(args.data_dir)
    photo_root = data_dir / args.photo_subdir
    sketch_root = data_dir / args.sketch_subdir

    photo_dataset = FolderDataset(
        photo_root,
        args.classes,
        args.max_size,
        args.max_samples_per_class,
    )
    sketch_dataset = FolderDataset(
        sketch_root,
        args.classes,
        args.max_size,
        args.max_samples_per_class,
    )

    device = torch.device(args.device)
    dataloader_kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": args.workers,
        "shuffle": False,
        "pin_memory": device.type == "cuda",
    }
    photo_loader = DataLoader(photo_dataset, **dataloader_kwargs)
    sketch_loader = DataLoader(sketch_dataset, **dataloader_kwargs)

    model, missing, unexpected = build_model(args, args.classes, device)
    photo_features, photo_classnames = extract_embeddings(
        model, photo_loader, args.classes, "photo", device
    )
    sketch_features, sketch_classnames = extract_embeddings(
        model, sketch_loader, args.classes, "sketch", device
    )

    all_features = torch.cat([photo_features, sketch_features], dim=0)
    coords, effective_perplexity = run_tsne(all_features, args.perplexity, args.seed)

    photo_coords = coords[: len(photo_features)]
    sketch_coords = coords[len(photo_features) :]
    cmap = plt.get_cmap("tab20", len(args.classes))
    colors = {
        classname: cmap(index)
        for index, classname in enumerate(args.classes)
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220, sharex=True, sharey=True)
    for ax in axes:
        ax.set_facecolor("#f5f4f1")
    fig.patch.set_facecolor("#f5f4f1")

    plot_domain(axes[0], photo_coords, photo_classnames, args.classes, colors, "Photo")
    plot_domain(axes[1], sketch_coords, sketch_classnames, args.classes, colors, "Sketch")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="upper right", frameon=True, fontsize=8)

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Checkpoint      : {ckpt_path}")
    print(f"Photo root      : {photo_root}")
    print(f"Sketch root     : {sketch_root}")
    print(f"Classes         : {', '.join(args.classes)}")
    print(f"Photo samples   : {len(photo_dataset)}")
    print(f"Sketch samples  : {len(sketch_dataset)}")
    print(f"Prompt depth    : {args.prompt_depth}")
    print(f"Context tokens  : {args.n_ctx}")
    print(f"t-SNE perplexity: {effective_perplexity}")
    print(f"Saved figure    : {output_path}")
    if missing:
        print(f"Missing keys    : {list(missing)}")
    if unexpected:
        print(f"Unexpected keys : {list(unexpected)}")


if __name__ == "__main__":
    main()
