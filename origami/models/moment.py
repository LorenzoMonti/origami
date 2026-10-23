"""MOMENT (Goswami et al. 2024) encoders, via the momentfm package.

MOMENT takes a fixed window of 512 points split into 64 patches of 8 points, with an
input mask (1 = observed). Shorter series are left-padded with zeros, as in the
package's own ClassificationDataset. A patch is observed for the model only if all
its 8 points are; masked patches get the learned [MASK] embedding and are excluded
from attention.
"""
import numpy as np

_MOMENT = dict(family="moment", license="MIT", extra="moment", context=512, patch=8, uses_time=False)

MODELS = {
    "moment-1-small": dict(_MOMENT, hf_id="AutonLab/MOMENT-1-small", params="38M"),
    "moment-1-base": dict(_MOMENT, hf_id="AutonLab/MOMENT-1-base", params="113M"),
    "moment-1-large": dict(_MOMENT, hf_id="AutonLab/MOMENT-1-large", params="346M"),
}


class Encoder:
    def __init__(self, spec, device=None):
        import torch
        from momentfm import MOMENTPipeline
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = MOMENTPipeline.from_pretrained(spec["hf_id"], model_kwargs={"task_name": "embedding"})
        model.init()
        self.model = model.to(self.device).float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode(self, values, masks, times, batch_size=256):
        """Masked mean of the last encoder layer's per-patch states (MOMENT has no special
        tokens). Curves are truncated to the last 512 points and left-padded with zeros.
        A patch counts as real if any of its points is real: the non-real points of such
        a patch are filled with 0 (the mean of the z-normalised curve) and marked observed
        for the model, so that its state encodes the real points instead of [MASK]."""
        import torch
        ctx, patch = self.spec["context"], self.spec["patch"]
        n_patch = ctx // patch
        out = []
        with torch.no_grad():
            for i in range(0, len(values), batch_size):
                s_b = [np.asarray(s, np.float32)[-ctx:] for s in values[i:i + batch_size]]
                m_b = [np.asarray(m, bool)[-ctx:] for m in masks[i:i + batch_size]]
                x = np.zeros((len(s_b), ctx), np.float32)
                real = np.zeros((len(s_b), ctx), bool)
                for j, (s, m) in enumerate(zip(s_b, m_b)):
                    x[j, ctx - len(s):] = np.where(m, np.nan_to_num(s), 0.0)
                    real[j, ctx - len(s):] = m
                keep = real.reshape(len(s_b), n_patch, patch).any(-1)  # (B, n_patch)
                input_mask = np.repeat(keep, patch, axis=1)
                x_t = torch.from_numpy(x).unsqueeze(1).to(self.device)  # (B, 1, 512)
                mask_t = torch.from_numpy(input_mask.astype(np.float32)).to(self.device)
                res = self.model.embed(x_enc=x_t, input_mask=mask_t, reduction="none")
                emb = res.embeddings[:, 0].float()  # (B, n_patch, d_model), single channel
                w = torch.from_numpy(keep).to(self.device).unsqueeze(-1).float()
                pooled = (emb * w).sum(1) / w.sum(1).clamp(min=1)
                out.append(pooled.cpu().numpy())
        return np.concatenate(out)
