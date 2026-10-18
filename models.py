from __future__ import annotations

import gzip
import logging
import random
import re
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

SEED = 42
log = logging.getLogger("experiments")

TRANSFORMERS = {
    "DistilBERT": "distilbert/distilbert-base-uncased",
    "BERT": "google-bert/bert-base-uncased",
    "RoBERTa": "FacebookAI/roberta-base",
}
ENCODER = "FacebookAI/roberta-base"

WORD_MAX_LEN = {"ISOT": 400}
SUBWORD_MAX_LEN = {"ISOT": 256}

W2V_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
           "word2vec-google-news-300/word2vec-google-news-300.gz")
TAU_QUANTILE = 0.90


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _lr(alpha: float) -> LogisticRegression:
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight={0: 1.0, 1: alpha})


class TfidfLR:
    config = {"ngram_range": [1, 2], "max_features": 30000, "stop_words": "english", "C": 1.0}

    def __init__(self, ctx):
        self.ctx = ctx

    def fit(self, splits, source, alpha):
        tr = splits[source]["train"]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, stop_words="english")
        self.clf = _lr(alpha).fit(self.vec.fit_transform(tr.text), tr.label)
        return {}

    def predict(self, splits, dataset):
        p = self.clf.predict_proba(self.vec.transform(splits[dataset]["test"].text))[:, 1]
        return (p > 0.5).astype(int), p


def train_net(net, train_data, val_data, collate, *, ctx, alpha, epochs, batch_size, lr,
              weight_decay=0.0, warmup=0.0, patience=None):
    dev = ctx.device
    net.to(dev)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=g)
    decay = [p for p in net.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in net.parameters() if p.requires_grad and p.ndim < 2]
    opt_cls = torch.optim.AdamW if weight_decay else torch.optim.Adam
    opt = opt_cls([{"params": decay, "weight_decay": weight_decay},
                   {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    sched = None
    if warmup:
        total, warm = epochs * len(loader), max(1, int(warmup * epochs * len(loader)))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(alpha), device=dev))
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    y_val = np.array([y for _, y in val_data])
    best = (-1.0, 0, None)
    for epoch in range(1, epochs + 1):
        net.train()
        running = 0.0
        for batch, y in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=amp):
                logits = net(**batch)
            loss = loss_fn(logits.float(), y.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
            running += loss.item() * len(y)
        f1 = f1_score(y_val, (predict_logits(net, val_data, collate, ctx) > 0).astype(int), zero_division=0)
        log.info("    epoch %d  loss %.4f  val F1 %.4f", epoch, running / len(train_data), f1)
        if f1 > best[0]:
            best = (f1, epoch, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        elif patience and epoch - best[1] >= patience:
            break
    net.load_state_dict(best[2])
    return {"best_epoch": best[1], "val_f1": round(best[0], 4)}


@torch.no_grad()
def predict_logits(net, data, collate, ctx, batch_size=128):
    net.eval()
    order = np.argsort([len(x) for x, _ in data], kind="stable")
    out = np.empty(len(data), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        batch, _ = collate([data[j] for j in idx])
        with torch.autocast(ctx.device.type, enabled=ctx.device.type == "cuda"):
            logits = net(**{k: v.to(ctx.device) for k, v in batch.items()})
        out[idx] = logits.float().cpu().numpy()
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


TOKEN = re.compile(r"[a-z0-9']+|[!?]")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def build_vocab(texts, size: int = 50000) -> list[str]:
    counts = Counter(t for text in texts for t in tokenize(text))
    return ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(size)]


def load_word2vec(path: str | None, wanted: set[str]) -> dict[str, np.ndarray]:
    if path is None:
        path = Path.home() / ".cache" / "word2vec" / "word2vec-google-news-300.gz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading word2vec Google News (1.7 GB) to %s", path)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(W2V_URL) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 24)
            tmp.rename(path)
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    vectors: dict[str, np.ndarray] = {}
    with opener(path, "rb") as f:
        n_words, dim = map(int, f.readline().split())
        rec = 4 * dim
        buf, pos = b"", 0
        for _ in range(n_words):
            sp = buf.find(b" ", pos)
            while sp < 0 or len(buf) < sp + 1 + rec:
                chunk = f.read(1 << 22)
                if not chunk:
                    raise EOFError(f"truncated word2vec file: {path}")
                buf, pos = buf[pos:] + chunk, 0
                sp = buf.find(b" ")
            word = buf[pos:sp].strip().decode("utf-8", errors="ignore")
            key = word.lower()
            if key in wanted and (word == key or key not in vectors):
                vectors[key] = np.frombuffer(buf, dtype=np.float32, count=dim, offset=sp + 1).copy()
            pos = sp + 1 + rec
    log.info("word2vec: %d of %d vocabulary words found", len(vectors), len(wanted))
    return vectors


def collate_words(items):
    seqs, labels = zip(*items)
    width = max(5, max(len(s) for s in seqs))
    ids = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    lengths = torch.tensor([len(s) for s in seqs])
    return {"ids": ids, "lengths": lengths}, torch.tensor(labels, dtype=torch.float32)


class TextCNNNet(nn.Module):
    def __init__(self, emb: np.ndarray, widths=(3, 4, 5), n_filters=100, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=False, padding_idx=0)
        self.widths = widths
        self.convs = nn.ModuleList(nn.Conv1d(emb.shape[1], n_filters, w) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * len(widths), 1)

    def forward(self, ids, lengths):
        x = self.emb(ids).transpose(1, 2)
        pooled = []
        for conv, w in zip(self.convs, self.widths):
            h = F.relu(conv(x))
            valid = (lengths - w + 1).clamp(min=1)
            pad = torch.arange(h.size(2), device=h.device)[None, :] >= valid[:, None]
            pooled.append(h.masked_fill(pad[:, None, :], 0.0).max(dim=2).values)
        return self.out(self.drop(torch.cat(pooled, dim=1))).squeeze(-1)


class BiLSTMNet(nn.Module):
    def __init__(self, emb: np.ndarray, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(emb.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.emb(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(self.drop(torch.cat([h[-2], h[-1]], dim=1))).squeeze(-1)


class WordModel:
    config = {"vocab": 50000, "max_len": {"ISOT": 400, "other": 64}, "embeddings": "word2vec-google-news-300",
              "dropout": 0.5, "optimizer": "Adam", "lr": 1e-3, "batch": 64, "max_epochs": 10, "patience": 2}

    def __init__(self, ctx, arch: str):
        self.ctx, self.arch = ctx, arch
        self.config = {**WordModel.config, **({"filters": [3, 4, 5], "n_filters": 100} if arch == "CNN"
                                               else {"hidden": 128, "bidirectional": True})}

    def _encode(self, texts, dataset):
        n = WORD_MAX_LEN.get(dataset, 64)
        return [[self.index.get(t, 1) for t in tokenize(x)][:n] or [1] for x in texts]

    def fit(self, splits, source, alpha):
        tr, va = splits[source]["train"], splits[source]["valid"]
        vocab = build_vocab(tr.text)
        self.index = {w: i for i, w in enumerate(vocab)}
        w2v = self.ctx.word2vec()
        rng = np.random.default_rng(SEED)
        emb = rng.uniform(-0.25, 0.25, (len(vocab), 300)).astype(np.float32)
        emb[0] = 0.0
        for w, i in self.index.items():
            if w in w2v:
                emb[i] = w2v[w]
        self.net = TextCNNNet(emb) if self.arch == "CNN" else BiLSTMNet(emb)
        train = list(zip(self._encode(tr.text, source), tr.label))
        valid = list(zip(self._encode(va.text, source), va.label))
        return train_net(self.net, train, valid, collate_words, ctx=self.ctx, alpha=alpha,
                         epochs=10, batch_size=64, lr=1e-3, patience=2)

    def predict(self, splits, dataset):
        te = splits[dataset]["test"]
        logits = predict_logits(self.net, list(zip(self._encode(te.text, dataset), te.label)), collate_words, self.ctx)
        return (logits > 0).astype(int), _sigmoid(logits)


def collate_subwords(pad_id):
    def collate(items):
        seqs, labels = zip(*items)
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}, torch.tensor(labels, dtype=torch.float32)
    return collate


class TransformerNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(self.drop(h)).squeeze(-1)


class FineTuned:
    config = {"max_len": {"ISOT": 256, "other": 64}, "truncation": "head", "optimizer": "AdamW", "lr": 2e-5,
              "weight_decay": 0.01, "batch": 16, "epochs": 3, "warmup": 0.1, "selection": "best val F1"}

    def __init__(self, ctx, key):
        from transformers import AutoTokenizer
        self.ctx, self.name = ctx, TRANSFORMERS[key]
        self.config = {**FineTuned.config, "checkpoint": self.name}
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.collate = collate_subwords(self.tok.pad_token_id)

    def _data(self, df, dataset):
        ids = self.tok(df.text.tolist(), truncation=True, max_length=SUBWORD_MAX_LEN.get(dataset, 64))["input_ids"]
        return list(zip(ids, df.label))

    def fit(self, splits, source, alpha):
        self.net = TransformerNet(self.name)
        return train_net(self.net, self._data(splits[source]["train"], source),
                         self._data(splits[source]["valid"], source), self.collate, ctx=self.ctx, alpha=alpha,
                         epochs=3, batch_size=16, lr=2e-5, weight_decay=0.01, warmup=0.1)

    def predict(self, splits, dataset):
        logits = predict_logits(self.net, self._data(splits[dataset]["test"], dataset), self.collate, self.ctx)
        return (logits > 0).astype(int), _sigmoid(logits)


class FrozenEncoder:
    def __init__(self, ctx, name=None):
        from transformers import AutoModel, AutoTokenizer
        self.ctx, self.name = ctx, name or ENCODER
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name).to(ctx.device).eval()
        self.collate = collate_subwords(self.tok.pad_token_id)
        self.cache: dict[tuple[str, str], np.ndarray] = {}

    @torch.no_grad()
    def __call__(self, splits, dataset, split, batch_size=64):
        if (dataset, split) not in self.cache:
            df = splits[dataset][split]
            ids = self.tok(df.text.tolist(), truncation=True, max_length=SUBWORD_MAX_LEN.get(dataset, 64))["input_ids"]
            order = np.argsort([len(x) for x in ids], kind="stable")
            out = np.empty((len(ids), self.model.config.hidden_size), dtype=np.float32)
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                batch, _ = self.collate([(ids[j], 0) for j in idx])
                batch = {k: v.to(self.ctx.device) for k, v in batch.items()}
                with torch.autocast(self.ctx.device.type, enabled=self.ctx.device.type == "cuda"):
                    h = self.model(**batch).last_hidden_state.float()
                m = batch["attention_mask"].unsqueeze(-1).float()
                out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
            self.cache[(dataset, split)] = out
            log.info("    encoded %s/%s: %d items", dataset, split, len(out))
        return self.cache[(dataset, split)]


class _FrozenFeatures:
    def _fit_pca(self, splits, source):
        H = self.ctx.encoder()(splits, source, "train")
        y = splits[source]["train"].label.to_numpy()
        self.pca = PCA(n_components=128, random_state=SEED).fit(H[y == 0])
        return self.pca.transform(H), y

    def _z(self, splits, dataset, split):
        return self.pca.transform(self.ctx.encoder()(splits, dataset, split))


class Probe(_FrozenFeatures):
    config = {"encoder": ENCODER, "pooling": "mean", "pca": 128, "pca_fit": "genuine source train",
              "classifier": "logistic regression", "C": 1.0}

    def __init__(self, ctx):
        self.ctx = ctx

    def fit(self, splits, source, alpha):
        z, y = self._fit_pca(splits, source)
        self.clf = _lr(alpha).fit(z, y)
        return {}

    def predict(self, splits, dataset):
        p = self.clf.predict_proba(self._z(splits, dataset, "test"))[:, 1]
        return (p > 0.5).astype(int), p


class OneClass(_FrozenFeatures):
    config = {"encoder": ENCODER, "pooling": "mean", "pca": 128, "pca_fit": "genuine source train",
              "covariance": "Ledoit-Wolf", "score": "squared Mahalanobis",
              "tau": "0.90 quantile of genuine source-validation scores"}

    def __init__(self, ctx):
        self.ctx = ctx

    def fit(self, splits, source, alpha):
        z, y = self._fit_pca(splits, source)
        self.cov = LedoitWolf().fit(z[y == 0])
        yv = splits[source]["valid"].label.to_numpy()
        self.val_scores = self.cov.mahalanobis(self._z(splits, source, "valid")[yv == 0])
        self.tau = float(np.quantile(self.val_scores, TAU_QUANTILE))
        return {"tau": round(self.tau, 4)}

    def predict(self, splits, dataset):
        s = self.cov.mahalanobis(self._z(splits, dataset, "test"))
        return (s > self.tau).astype(int), s


MODELS = {
    "LR": ("TF-IDF + Logistic Regression", lambda ctx: TfidfLR(ctx)),
    "CNN": ("Text CNN", lambda ctx: WordModel(ctx, "CNN")),
    "BiLSTM": ("BiLSTM", lambda ctx: WordModel(ctx, "BiLSTM")),
    "DistilBERT": ("DistilBERT (fine-tuned)", lambda ctx: FineTuned(ctx, "DistilBERT")),
    "BERT": ("BERT (fine-tuned)", lambda ctx: FineTuned(ctx, "BERT")),
    "RoBERTa": ("RoBERTa (fine-tuned)", lambda ctx: FineTuned(ctx, "RoBERTa")),
    "Probe": ("RoBERTa embeddings + LR (supervised probe)", lambda ctx: Probe(ctx)),
    "OneClass": ("One-class Mahalanobis (genuine only)", lambda ctx: OneClass(ctx)),
}


class Context:
    def __init__(self, splits, sources, w2v_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.splits, self.sources, self.w2v_path = splits, sources, w2v_path
        self._w2v, self._encoder = None, None

    def word2vec(self):
        if self._w2v is None:
            wanted = set().union(*(build_vocab(self.splits[s]["train"].text) for s in self.sources))
            self._w2v = load_word2vec(self.w2v_path, wanted)
        return self._w2v

    def encoder(self):
        if self._encoder is None:
            self._encoder = FrozenEncoder(self)
        return self._encoder
