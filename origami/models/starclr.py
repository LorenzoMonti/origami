"""StarCLR (Ding et al. 2026, arXiv:2604.24516): contrastive BERT-like encoder of light curves.

Frozen contrastive BACKBONE only (6 layers, 8 heads, d=512, 18.9M parameters, pretrained
on ~765k TESS light curves). The Zenodo release (doi:10.5281/zenodo.19728042, MIT) holds
one backbone, `checkpoints/backbone`, which the official inference code uses for every
survey; `checkpoints/{tess,ztf,gaia}` are supervised MLP classification heads on top of
the pooled backbone output (plus catalogue features), not adapted backbones, so they are
not used here.

Model definition: a pure-PyTorch re-implementation of `BertForPositionEmbeddingHiddenStates`
from github.com/dj-y/StarCLR-Inference at commit bedd60d3a1c4d57afa4a6395cbf7dbff1cde29ec
(src/infer/BertContrastiveLearningModel.py; MIT, Copyright (c) 2025 Junyao Ding; that file
derives from Hugging Face Transformers' BERT, Copyright 2018- The Hugging Face team,
Apache-2.0). The standard post-LayerNorm BERT encoder layers are written out here so that
the state-dict keys of the published safetensors load directly, without `transformers`
(whose BERT internals change across versions; upstream pins transformers 4.49.0).

Input protocol of the official preprocessing (src/infer/Preprocess.py, Gaia variant):
  token  = (t - t_min, scaled flux); no special tokens, no learned positions
  embed  = LayerNorm(Linear(1->512)(flux) + PE(t)),
           PE(t)[2i] = sin(t w_2i), PE(t)[2i+1] = cos(t w_2i+1), w_j = 2 pi / 10000^(j/512), t in days
  flux   = per curve: drop points with |f - mean| > 3 std (single pass), then
           (f - p5) / (p95 - p5) with the 5th/95th percentiles of the kept points
  output = mean of the last hidden states over the attended (real) tokens -> 512-d

Weights are downloaded on first use (the 116 MB Zenodo zip, MD5-checked; only the backbone
config and safetensors are kept, SHA-256-checked) into
  $ORIGAMI_WEIGHTS/starclr                      if ORIGAMI_WEIGHTS is set, else
  ${HF_HOME:-~/.cache/huggingface}/origami/starclr
After that no network access is needed; with HF_HUB_OFFLINE=1 a missing file is an error.
"""
import hashlib
import os

import numpy as np

_ZENODO_URL = "https://zenodo.org/api/records/19728042/files/StarCLR.zip/content"
_ZIP_MD5 = "ac0e95bdcd130b052e726d7c8b290813"
_FILES = {  # file kept in the cache -> (member of the zip, sha256)
    "config.json": ("StarCLR/checkpoints/backbone/config.json",
                    "2c05f446e0e8bc79be3bf931bbfb1731eca65aad05cf5f504797de0b30c57e83"),
    "model.safetensors": ("StarCLR/checkpoints/backbone/model.safetensors",
                          "6701a845091d75f9e7c12ef114ae13e81cdfb50eea91a3a3877fe7806beb0820"),
}

MODELS = {
    "starclr": dict(
        family="starclr", source="https://doi.org/10.5281/zenodo.19728042 (StarCLR.zip, checkpoints/backbone)",
        checksum="sha256:" + _FILES["model.safetensors"][1], zip_md5=_ZIP_MD5,
        code="https://github.com/dj-y/StarCLR-Inference@bedd60d3a1c4d57afa4a6395cbf7dbff1cde29ec",
        license="MIT (weights and code); model code partly derived from HF Transformers (Apache-2.0)",
        extra="starclr", install="pip install torch safetensors  (weights fetched from Zenodo on first use)",
        context=8192, params="18.9M", dim=512, layers=6, heads=8, uses_time=True, clip_sigma=3.0,
        trained_on="TESS, ~765k light curves (contrastive pretraining)"),
}


def weights_dir():
    root = os.environ.get("ORIGAMI_WEIGHTS")
    if root:
        return os.path.join(root, "starclr")
    hf = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf, "origami", "starclr")


def _digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights():
    """Path of the verified backbone directory, downloading it once if needed."""
    d = weights_dir()
    if all(os.path.exists(os.path.join(d, f)) for f in _FILES):
        for f, (_, sha) in _FILES.items():
            if _digest(os.path.join(d, f), "sha256") != sha:
                raise RuntimeError(f"checksum mismatch for {os.path.join(d, f)}; delete it to re-download")
        return d
    if os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False"):
        raise RuntimeError(f"StarCLR weights not in {d} and HF_HUB_OFFLINE is set")
    import tempfile
    import urllib.request
    import zipfile
    os.makedirs(d, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=d) as tmp:
        zpath = os.path.join(tmp, "StarCLR.zip")
        urllib.request.urlretrieve(_ZENODO_URL, zpath)
        if _digest(zpath, "md5") != _ZIP_MD5:
            raise RuntimeError("StarCLR.zip MD5 does not match the Zenodo record")
        with zipfile.ZipFile(zpath) as z:
            for f, (member, sha) in _FILES.items():
                out = os.path.join(tmp, f)
                with z.open(member) as src, open(out, "wb") as dst:  # fixed names: no path traversal
                    dst.write(src.read())
                if _digest(out, "sha256") != sha:
                    raise RuntimeError(f"SHA-256 mismatch for {member}")
        for f in _FILES:
            os.replace(os.path.join(tmp, f), os.path.join(d, f))
    return d


def preprocess(values, masks, times, context=8192, clip_sigma=3.0):
    """One curve -> (times, flux) float32 arrays fed to StarCLR.

    origami's `values` are z-normalised magnitudes; flux-like = -values (higher = brighter;
    a monotone but not linear stand-in for flux, the absolute scale is lost anyway).
    Real (mask) finite points only, the most recent `context` of them; then StarCLR's
    preprocessing: single-pass sigma clipping around the mean (np.std, ddof=0), min-max
    scaling between the 5th and 95th percentile of the kept points, times minus their minimum.
    """
    v, t = np.asarray(values, np.float64), np.asarray(times, np.float64)
    keep = np.asarray(masks, bool) & np.isfinite(v) & np.isfinite(t)
    f, t = -v[keep][-context:], t[keep][-context:]
    if len(f) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    s = f.std()
    if s > 0:
        ok = np.abs(f - f.mean()) <= clip_sigma * s
        f, t = f[ok], t[ok]
    lo, hi = np.percentile(f, 5), np.percentile(f, 95)
    f = (f - lo) / (hi - lo) if hi > lo else f - lo
    return (t - t.min()).astype(np.float32), f.astype(np.float32)


def _build(config):
    import math

    import torch
    from torch import nn
    import torch.nn.functional as F

    H, nh, eps = config["hidden_size"], config["num_attention_heads"], config["layer_norm_eps"]

    class Dense(nn.Module):  # BertSelfOutput / BertOutput: LayerNorm(x + dense(h))
        def __init__(self, n_in):
            super().__init__()
            self.dense = nn.Linear(n_in, H)
            self.LayerNorm = nn.LayerNorm(H, eps=eps)

        def forward(self, h, x):
            return self.LayerNorm(self.dense(h) + x)

    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query, self.key, self.value = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)

        def forward(self, x, keymask):  # keymask: (B, 1, 1, L) bool, True = attend
            B, L, _ = x.shape
            q, k, v = (m(x).view(B, L, nh, H // nh).transpose(1, 2) for m in (self.query, self.key, self.value))
            return F.scaled_dot_product_attention(q, k, v, attn_mask=keymask).transpose(1, 2).reshape(B, L, H)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self, self.output = SelfAttention(), Dense(H)

        def forward(self, x, keymask):
            return self.output(self.self(x, keymask), x)

    class Intermediate(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(H, config["intermediate_size"])

        def forward(self, x):
            return F.gelu(self.dense(x))  # "gelu" in HF = exact (erf) GELU

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention, self.intermediate = Attention(), Intermediate()
            self.output = Dense(config["intermediate_size"])

        def forward(self, x, keymask):
            a = self.attention(x, keymask)
            return self.output(self.intermediate(a), a)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList(Layer() for _ in range(config["num_hidden_layers"]))

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.toEmbedding = nn.Linear(1, H)
            self.LayerNorm = nn.LayerNorm(H, eps=eps)
            j = torch.arange(H, dtype=torch.float32)
            self.register_buffer("omega", 2 * math.pi / (10000 ** (j / H)), persistent=False)

        def forward(self, t, f):  # (B, L) float32 each
            x = self.toEmbedding(f.unsqueeze(-1))
            arg = t.unsqueeze(-1) * self.omega
            pe = torch.empty_like(x)
            pe[..., 0::2] = torch.sin(arg[..., 0::2])
            pe[..., 1::2] = torch.cos(arg[..., 1::2])
            return self.LayerNorm(x + pe)

    class Bert(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings, self.encoder = Embeddings(), Encoder()

    class StarCLR(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = Bert()

        def forward(self, t, f, mask):  # mask: (B, L) bool, True = real token
            x = self.bert.embeddings(t, f)
            keymask = mask[:, None, None, :]
            for layer in self.bert.encoder.layer:
                x = layer(x, keymask)
            w = mask.unsqueeze(-1).to(x.dtype)
            return (x * w).sum(1) / w.sum(1).clamp(min=1)

    return StarCLR()


class Encoder:
    def __init__(self, spec, device=None):
        import json

        import torch
        from safetensors.torch import load_file
        self.spec = spec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        d = fetch_weights()
        with open(os.path.join(d, "config.json")) as fh:
            config = json.load(fh)
        model = _build(config)
        model.load_state_dict(load_file(os.path.join(d, "model.safetensors")), strict=True)
        self.model = model.to(self.device).eval()
        self.dim = config["hidden_size"]

    def encode(self, values, masks, times, batch_size=256, max_attention=1 << 25):
        """Masked mean over real tokens of the last hidden states (StarCLR's own pooling; no
        special tokens exist). Curves are sorted by length and batched with right padding
        (padding is masked out of attention and of the mean); a batch is also split so that
        batch * length^2 <= max_attention. Curves with no real point give a zero vector."""
        import torch
        seqs = [preprocess(v, m, t, self.spec["context"], self.spec.get("clip_sigma", 3.0))
                for v, m, t in zip(values, masks, times)]
        out = np.zeros((len(seqs), self.dim), np.float32)
        order = [i for i in sorted(range(len(seqs)), key=lambda i: len(seqs[i][0])) if len(seqs[i][0])]
        batches, cur = [], []
        for i in order:  # increasing length: the last member sets the padded length
            L = len(seqs[i][0])
            if cur and (len(cur) >= batch_size or (len(cur) + 1) * L * L > max_attention):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        with torch.no_grad():
            for b in batches:
                L = max(len(seqs[i][0]) for i in b)
                t = torch.zeros((len(b), L))
                f = torch.zeros((len(b), L))
                m = torch.zeros((len(b), L), dtype=torch.bool)
                for j, i in enumerate(b):
                    n = len(seqs[i][0])
                    t[j, :n], f[j, :n], m[j, :n] = torch.from_numpy(seqs[i][0]), torch.from_numpy(seqs[i][1]), True
                emb = self.model(t.to(self.device), f.to(self.device), m.to(self.device))
                out[b] = emb.float().cpu().numpy()
        return out
