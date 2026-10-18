from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATASETS = ("ISOT", "LIAR", "PolitiFact", "GossipCop")

LIAR_FAKE = {"pants-fire", "false", "barely-true"}
LIAR_REAL = {"half-true", "mostly-true", "true"}

REUTERS_TAG = re.compile(r"\(Reuters\)\s*-\s*")
WS = re.compile(r"\s+")


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(lambda x: WS.sub(" ", x).strip())


def _find(data_dir: Path, folder: str, name: str) -> Path:
    base = data_dir / folder if (data_dir / folder).is_dir() else data_dir
    hits = sorted(base.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {base}")
    return hits[0]


def _drop_conflicts_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    conflict = df.groupby("text").label.nunique().gt(1)
    df = df[~df.text.isin(conflict[conflict].index)]
    return df.drop_duplicates("text")


def load_isot(data_dir: Path) -> pd.DataFrame:
    fake = pd.read_csv(_find(data_dir, "ISOT", "Fake.csv")).assign(label=1)
    real = pd.read_csv(_find(data_dir, "ISOT", "True.csv")).assign(label=0)
    df = pd.concat([fake, real], ignore_index=True)
    df = df[_norm(df["text"]).ne("")].copy()
    df["text"] = _norm(df["title"] + " " + df["text"].str.replace(REUTERS_TAG, "", regex=True))
    return _drop_conflicts_and_duplicates(df)[["text", "label"]].reset_index(drop=True)


def load_liar(data_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for split in ("train", "valid", "test"):
        d = pd.read_csv(_find(data_dir, "LIAR", f"{split}.tsv"), sep="\t", header=None, quoting=3)
        d["text"] = _norm(d[2])
        d["label"] = d[1].map(lambda l: 1 if l in LIAR_FAKE else (0 if l in LIAR_REAL else None))
        out[split] = d.dropna(subset=["label"]).astype({"label": int})[["text", "label"]]
    test_texts = set(out["test"].text)
    for split in ("train", "valid"):
        out[split] = out[split][~out[split].text.isin(test_texts)]
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def load_fakenewsnet(data_dir: Path) -> dict[str, pd.DataFrame]:
    subsets = {}
    for name in ("politifact", "gossipcop"):
        f = pd.read_csv(_find(data_dir, "FakeNewsNet", f"{name}_fake.csv")).assign(label=1)
        r = pd.read_csv(_find(data_dir, "FakeNewsNet", f"{name}_real.csv")).assign(label=0)
        d = pd.concat([f, r], ignore_index=True)
        d["text"] = _norm(d["title"])
        subsets[name] = _drop_conflicts_and_duplicates(d[d.text.ne("")])[["text", "label"]]
    shared = set(subsets["politifact"].text) & set(subsets["gossipcop"].text)
    return {k: v[~v.text.isin(shared)].reset_index(drop=True) for k, v in subsets.items()}


def split_isot(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    trval, test = train_test_split(df, test_size=0.25, stratify=df.label, random_state=SEED)
    train, valid = train_test_split(trval, test_size=0.10, stratify=trval.label, random_state=SEED)
    return {"train": train, "valid": valid, "test": test}


def split_702010(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    valid, test = train_test_split(rest, test_size=2 / 3, stratify=rest.label, random_state=SEED)
    return {"train": train, "valid": valid, "test": test}


def load_splits(data_dir: str | Path, datasets=DATASETS) -> dict[str, dict[str, pd.DataFrame]]:
    data_dir = Path(data_dir)
    splits = {}
    if "ISOT" in datasets:
        splits["ISOT"] = split_isot(load_isot(data_dir))
    if "LIAR" in datasets:
        splits["LIAR"] = load_liar(data_dir)
    if "PolitiFact" in datasets or "GossipCop" in datasets:
        fnn = load_fakenewsnet(data_dir)
        for name, key in (("PolitiFact", "politifact"), ("GossipCop", "gossipcop")):
            if name in datasets:
                splits[name] = split_702010(fnn[key])
    return {ds: {k: v.reset_index(drop=True) for k, v in sp.items()} for ds, sp in splits.items()}
