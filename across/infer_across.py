import argparse
import json
import sys
import types
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

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
    return ((torch.arange(len(positions), device=positions.device, dtype=torch.float32) + 1) / positions).mean()


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
    torchmetrics_functional_stub.retrieval_average_precision = _fallback_retrieval_average_precision
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

from across.data_config_across import ACROSS_DATASET_CLASSES
from src.model import ZS_SBIR
from src.sketchy_dataset import normal_transform
from src.utils import retrieval_average_precision, retrieval_precision


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cross-dataset SBIR inference for a sketchy_1 checkpoint."
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["tuberlin", "quickdraw"],
        choices=sorted(ACROSS_DATASET_CLASSES.keys()),
        help="Target datasets to evaluate",
    )
    parser.add_argument(
        "--base_root",
        type=str,
        default="",
        help="Base directory containing <dataset>/photo and <dataset>/sketch",
    )
    parser.add_argument(
        "--root_template",
        type=str,
        default="",
        help="Optional template such as /kaggle/input/datasets/.../{dataset}",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Single dataset root containing photo/ and sketch/",
    )
    parser.add_argument("--photo_subdir", type=str, default="photo")
    parser.add_argument("--sketch_subdir", type=str, default="sketch")
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
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--precision_at", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save metrics as JSON",
    )

    # Kept for compatibility with the model config.
    parser.add_argument("--use_adapt_sk", type=bool, default=True)
    parser.add_argument("--use_adapt_ph", type=bool, default=True)
    parser.add_argument("--use_adapt_txt", type=bool, default=True)
    parser.add_argument("--use_co_sk", type=bool, default=True)
    parser.add_argument("--use_co_ph", type=bool, default=True)
    parser.add_argument("--progress", type=bool, default=False)
    parser.add_argument("--gzs", type=bool, default=False)
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_classes", type=int, default=104)
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default="across_inference")
    return parser.parse_args()


def canonicalize_class_name(name):
    return name.replace("_", " ").replace("-", " ").strip().lower()


def resolve_dataset_root(args, dataset_name):
    if args.root:
        if len(args.datasets) != 1:
            raise ValueError("--root can only be used when a single dataset is passed to --datasets.")
        return Path(args.root)

    if args.root_template:
        return Path(args.root_template.format(dataset=dataset_name))

    if args.base_root:
        return Path(args.base_root) / dataset_name

    raise ValueError("Please provide one of --root, --root_template, or --base_root.")


class AcrossEvalDataset(Dataset):
    def __init__(self, root, split, classnames, max_size):
        self.root = Path(root)
        self.split = split
        self.classnames = list(classnames)
        self.max_size = max_size
        self.transform = normal_transform()
        self.samples = []
        self.resolved_dirs = {}

        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")

        available_dirs = {
            canonicalize_class_name(path.name): path.name
            for path in split_root.iterdir()
            if path.is_dir()
        }

        missing_classes = []
        for label, classname in enumerate(self.classnames):
            class_dir_name = self._resolve_dir_name(classname, available_dirs)
            if class_dir_name is None:
                missing_classes.append(classname)
                continue

            class_dir = split_root / class_dir_name
            image_paths = sorted(path for path in class_dir.iterdir() if path.is_file())
            if not image_paths:
                missing_classes.append(classname)
                continue

            self.resolved_dirs[classname] = class_dir_name
            self.samples.extend((path, label) for path in image_paths)

        if missing_classes:
            raise FileNotFoundError(
                f"Missing or empty class folders under {split_root}: {missing_classes}"
            )

        if not self.samples:
            raise RuntimeError(f"No samples found under {split_root}")

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
        image_path, label = self.samples[index]
        image = ImageOps.pad(
            Image.open(image_path).convert("RGB"),
            size=(self.max_size, self.max_size),
        )
        return self.transform(image), label


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


def build_model(args, classnames, device):
    model = ZS_SBIR(args=args, classname=classnames)
    state_dict = load_checkpoint_state(args.ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if device.type == "cpu":
        model.float()
    return model, missing, unexpected


def extract_features(model, dataloader, classnames, image_type, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            batch_features = model.model.extract_feature(
                images,
                classname=classnames,
                type=image_type,
            )
            features.append(batch_features.detach().cpu())
            labels.append(batch_labels.detach().cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def compute_retrieval_metrics(
    query_features,
    query_labels,
    gallery_features,
    gallery_labels,
    classnames,
    precision_at,
):
    ap_scores = torch.zeros(len(query_features), dtype=torch.float32)
    precision_scores = torch.zeros(len(query_features), dtype=torch.float32)

    for idx, query_feature in enumerate(query_features):
        similarity = F.cosine_similarity(query_feature.unsqueeze(0), gallery_features)
        target = gallery_labels == query_labels[idx]
        ap_scores[idx] = retrieval_average_precision(similarity, target)
        precision_scores[idx] = retrieval_precision(similarity, target, top_k=precision_at)

    class_ranking = []
    precision_key = f"P@{precision_at}"
    for label_idx, classname in enumerate(classnames):
        mask = query_labels == label_idx
        if not mask.any():
            continue

        class_ranking.append(
            {
                "class": classname,
                "mAP": ap_scores[mask].mean().item(),
                precision_key: precision_scores[mask].mean().item(),
                "num_queries": int(mask.sum().item()),
            }
        )

    class_ranking.sort(
        key=lambda item: (item["mAP"], item[precision_key], item["class"]),
        reverse=True,
    )

    return ap_scores.mean().item(), precision_scores.mean().item(), class_ranking


def evaluate_one_dataset(args, dataset_name, device):
    dataset_root = resolve_dataset_root(args, dataset_name)
    classnames = ACROSS_DATASET_CLASSES[dataset_name]

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    args.dataset = dataset_name

    sketch_dataset = AcrossEvalDataset(
        root=dataset_root,
        split=args.sketch_subdir,
        classnames=classnames,
        max_size=args.max_size,
    )
    photo_dataset = AcrossEvalDataset(
        root=dataset_root,
        split=args.photo_subdir,
        classnames=classnames,
        max_size=args.max_size,
    )

    dataloader_kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": args.workers,
        "shuffle": False,
        "pin_memory": device.type == "cuda",
    }
    sketch_loader = DataLoader(sketch_dataset, **dataloader_kwargs)
    photo_loader = DataLoader(photo_dataset, **dataloader_kwargs)

    model, missing, unexpected = build_model(args, classnames, device)
    query_features, query_labels = extract_features(model, sketch_loader, classnames, "sketch", device)
    gallery_features, gallery_labels = extract_features(model, photo_loader, classnames, "photo", device)

    mAP_all, precision, class_ranking = compute_retrieval_metrics(
        query_features=query_features,
        query_labels=query_labels,
        gallery_features=gallery_features,
        gallery_labels=gallery_labels,
        classnames=classnames,
        precision_at=args.precision_at,
    )

    return {
        "dataset": dataset_name,
        "root": str(dataset_root),
        "classes": classnames,
        "num_classes": len(classnames),
        "num_queries": len(sketch_dataset),
        "num_gallery": len(photo_dataset),
        "mAP_all": mAP_all,
        f"P@{args.precision_at}": precision,
        "class_ranking": class_ranking,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    results = []
    for dataset_name in args.datasets:
        result = evaluate_one_dataset(args, dataset_name, device)
        results.append(result)

        print(f"[{dataset_name}] root={result['root']}")
        print(
            f"[{dataset_name}] mAP@all={result['mAP_all']:.4f}, "
            f"P@{args.precision_at}={result[f'P@{args.precision_at}']:.4f}, "
            f"queries={result['num_queries']}, gallery={result['num_gallery']}"
        )
        print(f"[{dataset_name}] class ranking (desc by class-wise mAP):")
        for rank, item in enumerate(result["class_ranking"], start=1):
            print(
                f"  {rank:02d}. {item['class']} | "
                f"mAP={item['mAP']:.4f} | "
                f"P@{args.precision_at}={item[f'P@{args.precision_at}']:.4f} | "
                f"queries={item['num_queries']}"
            )
        if result["missing_keys"] or result["unexpected_keys"]:
            print(f"[{dataset_name}] missing_keys={result['missing_keys']}")
            print(f"[{dataset_name}] unexpected_keys={result['unexpected_keys']}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
