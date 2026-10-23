"""
Frozen foundation-model embedders.

An Embedder turns a LightCurves set into one vector per curve: the input
transform builds the sequences, the model plugin (origami.models) encodes them
and pools the states over the positions that come from real data. Weights are
downloaded on first use; results are cached on disk, keyed by model,
transform, transform code and data fingerprint.

Environment variables:
  ORIGAMI_CACHE    embedding cache directory (default ~/.cache/origami)
  HF_HUB_OFFLINE=1 use only weights already downloaded (e.g. compute nodes
                   without internet; run origami.prefetch on a login node)
"""
import hashlib
import inspect
import json
import os

import numpy as np

from . import models, transforms

# cached embeddings are invalidated whenever the transforms module changes
_TRANSFORMS_HASH = hashlib.sha256(inspect.getsource(transforms).encode()).hexdigest()[:12]


def available_models():
    return {name: dict(spec) for name, spec in models.registry().items()}


def cache_dir():
    return os.environ.get("ORIGAMI_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "origami"))


class MissingEmbedding(RuntimeError):
    """Raised when an embedding is not on disk and computing it is not allowed."""


class Embedder:
    """Frozen encoder + input transform + masked mean pooling.

    cache_dir: where embeddings are stored (default: ORIGAMI_CACHE or ~/.cache/origami)
    compute:   if False, only read stored embeddings and raise MissingEmbedding otherwise
               (used by `origami collect`, which must not run models)
    """

    def __init__(self, model, input="folded-grid", device=None, batch_size=256, cache=True, cache_dir=None,
                 compute=True, **transform_kwargs):
        if model not in models.registry():
            raise ValueError(f"unknown model {model!r}; available: {sorted(models.registry())}")
        if input not in transforms.TRANSFORMS:
            raise ValueError(f"unknown input transform {input!r}; available: {sorted(transforms.TRANSFORMS)}")
        self.model, self.input, self.spec = model, input, models.registry()[model]
        self.device, self.batch_size, self.cache = device, batch_size, cache
        self.cache_root, self.compute = cache_dir, compute
        self.transform_kwargs = transform_kwargs
        self._encoder = None

    @property
    def name(self):
        return f"{self.model}[{self.input}]"

    def transform(self, lcs):
        path = self._cache_path(lcs)
        if self.cache and os.path.exists(path):
            return np.load(path)
        if not self.compute:
            raise MissingEmbedding(f"{self.name}: no stored embedding for this data ({path})")
        seq = transforms.apply(self.input, lcs, **self.transform_kwargs)
        encoder = self._load()
        extra = {"scales": seq.scales} if "scales" in inspect.signature(encoder.encode).parameters else {}
        emb = encoder.encode(seq.values, seq.masks, seq.times, batch_size=self.batch_size, **extra)
        if self.cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, emb)
        return emb

    def _cache_path(self, lcs):
        spec = {k: v for k, v in self.spec.items() if k != "module"}  # e.g. a changed layer invalidates the cache
        key = json.dumps([self.model, spec, self.input, self.transform_kwargs, _TRANSFORMS_HASH, lcs.fingerprint()],
                         sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        return os.path.join(self.cache_root or cache_dir(), self.model, f"{self.input}_{digest}.npy")

    def _load(self):
        if self._encoder is None:
            self._encoder = models.get_encoder(self.model, device=self.device)
        return self._encoder


def prefetch(names):
    """Download weights in advance (run where internet is available)."""
    for name in names:
        Embedder(name)._load()
