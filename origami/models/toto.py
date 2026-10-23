"""Toto 2.0 (Khwaja et al. 2026, Datadog) encoders, via the toto-models package (import name toto2).

Toto 2.0 is a decoder-only patched transformer (patch = 32 points) that alternates causal
time-axis attention with full variate-axis attention; the series is fed as one variate.
The public API only returns quantile forecasts: the embedding is taken from the backbone
(`Toto2Model.transformer`, a VariateTimeTransformerDecoder, output after its final RMSNorm),
captured with a forward hook, i.e. the per-patch states that feed the quantile head.
Timestamps are not used (positions are integer patch indices with xPos rotary embeddings),
so `times` is ignored. Weights: Apache-2.0. toto2 requires Python >= 3.12.
"""
import numpy as np

# context: 4096 points = 128 patches, inferred from residual_attn_ratio = sqrt(S / log S) with
# S = 128 in every config.json (the model card does not state a context length).
_T2 = dict(family="toto-2", license="Apache-2.0", extra="toto", context=4096, patch=32, uses_time=False,
           requires_python="3.12")

TOKEN_BUDGET = 16384  # max batch x patches per forward pass (memory guard)

MODELS = {
    "toto-2.0-4m": dict(_T2, hf_id="Datadog/Toto-2.0-4m", params="4M", dim=256),
    "toto-2.0-22m": dict(_T2, hf_id="Datadog/Toto-2.0-22m", params="22M", dim=512),
    "toto-2.0-313m": dict(_T2, hf_id="Datadog/Toto-2.0-313m", params="313M", dim=1024),
    "toto-2.0-1b": dict(_T2, hf_id="Datadog/Toto-2.0-1B", params="1B", dim=1536),
    "toto-2.0-2.5b": dict(_T2, hf_id="Datadog/Toto-2.0-2.5B", params="2.5B", dim=2048),
}


class Encoder:
    def __init__(self, spec, device=None):
        import torch
        from toto2 import Toto2Model
        self.spec = spec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Toto2Model.from_pretrained(spec["hf_id"]).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.model.config.patch_size != spec["patch"]:
            raise ValueError(f"{spec['hf_id']} has patch size {self.model.config.patch_size}, spec says {spec['patch']}")
        self._states = None
        self.model.transformer.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        self._states = output

    def encode(self, values, masks, times, batch_size=256):
        """Masked mean of the backbone's final-layer patch states. Each curve (last `context`
        points) is left-padded up to the batch's longest curve rounded up to a multiple of the
        patch size; padding and masked points are passed with target_mask False and value 0
        (as the GluonTS wrapper does: it pads on the left). Patches with no observed point get
        group id -1 inside the model, which isolates them from the real patches in the causal
        time attention, and the causal scaler ignores unobserved points, so the states of the
        real patches do not depend on the padding. A patch is pooled if any of its points is
        real. There are no special tokens. Batches are split so that a forward pass holds at
        most TOKEN_BUDGET patches."""
        patch, ctx_max = self.spec["patch"], self.spec["context"]
        s_all = [np.asarray(s, np.float32)[-ctx_max:] for s in values]
        m_all = [np.asarray(m, bool)[-ctx_max:] for m in masks]
        out = []
        for i in range(0, len(s_all), batch_size):
            T = max(-(-len(s) // patch) for s in s_all[i:i + batch_size])
            sub = max(1, min(batch_size, TOKEN_BUDGET // T))
            for k in range(i, min(i + batch_size, len(s_all)), sub):
                j = min(k + sub, i + batch_size, len(s_all))
                out.append(self._encode_batch(s_all[k:j], m_all[k:j]))
        return np.concatenate(out)

    def _encode_batch(self, s_b, m_b):
        import torch
        patch = self.spec["patch"]
        L = -(-max(len(s) for s in s_b) // patch) * patch
        target = torch.zeros((len(s_b), 1, L), dtype=torch.float32)
        mask = torch.zeros((len(s_b), 1, L), dtype=torch.bool)
        for j, (s, m) in enumerate(zip(s_b, m_b)):
            m = m & np.isfinite(s)
            target[j, 0, L - len(s):] = torch.from_numpy(np.where(m, s, 0).astype(np.float32))
            mask[j, 0, L - len(s):] = torch.from_numpy(m)
        target, mask = target.to(self.device), mask.to(self.device)
        series_ids = torch.zeros((len(s_b), 1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            self._states = None
            # cpm_mask all True: no contiguous patch masking, every patch is context
            self.model(target, mask, torch.ones_like(mask), series_ids, num_return_steps=1)
            states = self._states[:, 0].float()  # (batch, n_patches, d_model)
            self._states = None
        pm = mask[:, 0].view(len(s_b), L // patch, patch).any(-1).unsqueeze(-1).float()
        return ((states * pm).sum(1) / pm.sum(1).clamp(min=1)).cpu().numpy()
