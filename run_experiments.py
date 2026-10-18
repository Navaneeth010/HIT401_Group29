from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import SEED, load_splits
from models import MODELS, Context, set_seed

SETTINGS = {
    "S1": ("ISOT", "ISOT", "in-domain"),
    "S2": ("ISOT", "LIAR", "cross-dataset"),
    "S3": ("LIAR", "LIAR", "target reference"),
    "S4": ("PolitiFact", "PolitiFact", "in-domain"),
    "S5": ("GossipCop", "PolitiFact", "cross-topic"),
    "S6": ("GossipCop", "GossipCop", "in-domain"),
    "S7": ("PolitiFact", "GossipCop", "cross-topic"),
}
SWEEP_QUANTILES = np.round(np.arange(0.80, 0.99501, 0.005), 3)

log = logging.getLogger("experiments")


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    out = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp + tn) / n, "precision": precision,
           "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "specificity": specificity,
           "balanced_accuracy": (recall + specificity) / 2, "pred_fake_rate": (tp + fp) / n}
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in out.items()}


def upsert(path: Path, rows: list[dict], keys: list[str]) -> None:
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.set_index(keys).index.isin(new.set_index(keys).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def describe(row_id, setting, kind, train, test, key, name, y) -> dict:
    return {"row_id": row_id, "setting": setting, "setting_type": kind, "train_domain": train,
            "test_domain": test, "model_key": key, "model": name, "n": len(y), "n_fake": int(y.sum()),
            "n_real": int(len(y) - y.sum())}


def parse_args():
    p = argparse.ArgumentParser(description="Run the cross-domain fake news detection experiments (settings S1-S7, eight models).")
    p.add_argument("--data-dir", default="data", help="folder that contains ISOT/, LIAR/ and FakeNewsNet/")
    p.add_argument("--out", default="results", help="output folder")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    p.add_argument("--w2v-path", help="GoogleNews word2vec .bin or .gz file (downloaded if omitted)")
    p.add_argument("--rerun", action="store_true", help="repeat experiments already in metrics.csv")
    p.add_argument("--max-items", type=int, help="subsample every split to this size (functional test only)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run_log.txt", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    import sklearn
    import transformers
    log.info("python %s | torch %s | transformers %s | scikit-learn %s | numpy %s | pandas %s",
             platform.python_version(), torch.__version__, transformers.__version__, sklearn.__version__,
             np.__version__, pd.__version__)
    log.info("device: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    log.info("arguments: %s", vars(args))

    settings = [s for s in SETTINGS if s in args.settings]
    sources = list(dict.fromkeys(SETTINGS[s][0] for s in settings))
    targets = list(dict.fromkeys(SETTINGS[s][1] for s in settings))
    splits = load_splits(args.data_dir, set(sources) | set(targets))
    if args.max_items:
        log.warning("--max-items %d: splits are subsampled; results are for testing the code only", args.max_items)
        splits = {ds: {k: v.sample(n=min(args.max_items, len(v)), random_state=SEED).reset_index(drop=True)
                       for k, v in sp.items()} for ds, sp in splits.items()}
    for ds, sp in splits.items():
        log.info("%-10s %s", ds, "  ".join(f"{k} {len(v)} ({v.label.mean():.1%} fake)" for k, v in sp.items()))

    metrics_csv, sweep_csv = out / "metrics.csv", out / "threshold_sweep.csv"
    done = set() if args.rerun or not metrics_csv.exists() else set(pd.read_csv(metrics_csv).row_id)

    rows = []
    for ds in targets:
        y = splits[ds]["test"].label.to_numpy()
        for key, name, value in (("Majority", "Majority class (always genuine)", 0), ("AlwaysFake", "Always fake", 1)):
            rows.append({**describe(f"{ds}-{key}", f"test:{ds}", "reference rule", "-", ds, key, name, y),
                         **scores(y, np.full_like(y, value)), "seed": SEED})
    upsert(metrics_csv, rows, ["row_id"])

    ctx = Context(splits, sources, args.w2v_path)
    for key in (m for m in MODELS if m in args.models):
        name, build = MODELS[key]
        for src in sources:
            todo = [s for s in settings if SETTINGS[s][0] == src and f"{s}-{key}" not in done]
            if not todo:
                continue
            labels = splits[src]["train"].label
            alpha = float((labels == 0).sum() / (labels == 1).sum()) if src == "GossipCop" else 1.0
            log.info("%s trained on %s (fake-class weight %.3f) -> %s", key, src, alpha, ", ".join(todo))
            set_seed(SEED)
            model = build(ctx)
            start = time.time()
            info = model.fit(splits, src, alpha)
            seconds = round(time.time() - start, 1)
            config = {**model.config, **({"max_items": args.max_items} if args.max_items else {})}
            rows, sweep = [], []
            for s in todo:
                tgt = SETTINGS[s][1]
                y = splits[tgt]["test"].label.to_numpy()
                pred, score = model.predict(splits, tgt)
                m = scores(y, pred)
                log.info("  %s-%-10s acc %.4f  P %.4f  R %.4f  F1 %.4f  BA %.4f", s, key, m["accuracy"],
                         m["precision"], m["recall"], m["f1"], m["balanced_accuracy"])
                rows.append({**describe(f"{s}-{key}", s, SETTINGS[s][2], src, tgt, key, name, y), **m,
                             "threshold": round(getattr(model, "tau", 0.5), 6), "best_epoch": info.get("best_epoch"),
                             "val_f1": info.get("val_f1"), "fake_weight": round(alpha, 4), "train_seconds": seconds,
                             "seed": SEED, "config": json.dumps(config)})
                if key == "OneClass":
                    for q in SWEEP_QUANTILES:
                        tau = float(np.quantile(model.val_scores, q))
                        sweep.append({"setting": s, "train_domain": src, "test_domain": tgt, "tau_quantile": q,
                                      "tau": round(tau, 6), **scores(y, (score > tau).astype(int))})
            upsert(metrics_csv, rows, ["row_id"])
            if sweep:
                upsert(sweep_csv, sweep, ["setting", "tau_quantile"])
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log.info("finished; results in %s", metrics_csv.resolve())


if __name__ == "__main__":
    main()
