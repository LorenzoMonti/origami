"""Chronos (Ansari et al. 2024), Chronos-Bolt and Chronos-2 encoders, via the chronos-forecasting package."""
import numpy as np

_T5 = dict(family="chronos", license="Apache-2.0", extra="chronos", context=512, uses_time=False)
_BOLT = dict(family="chronos-bolt", license="Apache-2.0", extra="chronos", context=2048, patch=16, uses_time=False)
_C2 = dict(family="chronos-2", license="Apache-2.0", extra="chronos", context=8192, patch=16, uses_time=False)

MODELS = {
    "chronos-t5-tiny": dict(_T5, hf_id="amazon/chronos-t5-tiny", params="8M"),
    "chronos-t5-mini": dict(_T5, hf_id="amazon/chronos-t5-mini", params="20M"),
    "chronos-t5-small": dict(_T5, hf_id="amazon/chronos-t5-small", params="46M"),
    "chronos-t5-base": dict(_T5, hf_id="amazon/chronos-t5-base", params="200M"),
    "chronos-bolt-tiny": dict(_BOLT, hf_id="amazon/chronos-bolt-tiny", params="9M"),
    "chronos-bolt-small": dict(_BOLT, hf_id="amazon/chronos-bolt-small", params="48M"),
    "chronos-2-small": dict(_C2, hf_id="autogluon/chronos-2-small", params="28M"),
    "chronos-2": dict(_C2, hf_id="amazon/chronos-2", params="120M"),
}


class Encoder:
    def __init__(self, spec, device=None):
        import torch
        from chronos import BaseChronosPipeline
        self.spec = spec
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = BaseChronosPipeline.from_pretrained(spec["hf_id"], device_map=device, torch_dtype=torch.float32)

    def encode(self, values, masks, times, batch_size=256):
        if self.spec["family"] == "chronos-2":
            return self._encode_chronos2(values, masks, batch_size)
        return self._encode_t5_bolt(values, masks, batch_size)

    def _encode_chronos2(self, values, masks, batch_size):
        """Chronos-2 returns, per series, (n_variates, n_patches + 2, d): left-padded context patches,
        then [REG], then one output patch. Masked mean over the patches that hold real points."""
        import torch
        ctx_max, patch = self.spec["context"], self.spec["patch"]
        values = [v[-ctx_max:] for v in values]
        masks = [m[-ctx_max:] for m in masks]
        emb, _ = self.pipeline.embed([torch.from_numpy(v) for v in values], batch_size=batch_size)
        out = []
        for e, v, m in zip(emb, values, masks):
            n_real = -(-len(v) // patch)
            states = e[0, -(n_real + 2):-2].float()  # this series' patches, special tokens dropped
            pad = n_real * patch - len(v)
            pm = torch.from_numpy(np.r_[np.zeros(pad, bool), m]).view(n_real, patch).any(-1)
            out.append(states[pm].mean(0).cpu().numpy())
        return np.stack(out)

    def _encode_t5_bolt(self, values, masks, batch_size):
        """Masked mean of encoder states. T5: one state per point, EOS dropped. Bolt: one state
        per patch of 16 points (a patch counts if any of its points is real), [REG] dropped."""
        import torch
        ctx_max, patch = self.spec["context"], self.spec.get("patch")
        out = []
        with torch.no_grad():
            for i in range(0, len(values), batch_size):
                s_b = [s[-ctx_max:] for s in values[i:i + batch_size]]
                m_b = [m[-ctx_max:] for m in masks[i:i + batch_size]]
                L = max(len(s) for s in s_b)
                ctx = torch.full((len(s_b), L), float("nan"))
                msk = torch.zeros((len(s_b), L), dtype=torch.bool)
                for j, (s, m) in enumerate(zip(s_b, m_b)):
                    ctx[j, L - len(s):] = torch.from_numpy(s)
                    msk[j, L - len(s):] = torch.from_numpy(m)
                emb, _ = self.pipeline.embed(ctx)
                emb = emb.float().cpu()
                if patch:  # the model left-pads to a multiple of the patch size
                    n_patch = -(-L // patch)
                    pad = n_patch * patch - L
                    msk = torch.cat([torch.zeros((len(s_b), pad), dtype=torch.bool), msk], 1)
                    msk = msk.view(len(s_b), n_patch, patch).any(-1)
                    emb = emb[:, :n_patch]
                else:
                    emb = emb[:, :L]
                w = msk.unsqueeze(-1).float()
                out.append(((emb * w).sum(1) / w.sum(1).clamp(min=1)).numpy())
        return np.concatenate(out)
