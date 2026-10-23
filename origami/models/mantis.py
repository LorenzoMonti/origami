"""Mantis (Feofanov et al. 2025, 2026) time-series classification foundation models, via mantis-tsfm.

Mantis takes a fixed-length univariate series (pre-trained on length 512, 32 patches of 16
points) and returns one vector per series; there are no per-position states to pool. Following
the authors' recommendation, every series is resized to 512 points by linear interpolation
(torch.nn.functional.interpolate, mode="linear", align_corners=False) rather than padded.

Masks are honoured when building the input: only real positions (mask True), in their original
order, are kept and then resized, so padding never enters the model. Timestamps are not used:
the resizing is over the position index, as in the official recipe, so gaps in irregularly
sampled (raw) curves are ignored.

Embedding: the authors report that, for a frozen encoder, an intermediate transformer layer with
the "combined" output (CLS token concatenated with the mean of the 32 patch tokens) works best
on UCR; `layer` is that layer (0-based) as listed in the mantis-tsfm README. The "-cls" variant
is the package default (CLS token of the last layer).
"""
import numpy as np

_BASE = dict(license="Apache-2.0", extra="mantis", context=512, seq_len=512, uses_time=False)

MODELS = {
    "mantis-8m": dict(_BASE, family="mantis", arch="v1", hf_id="paris-noah/Mantis-8M", params="8M",
                      layer=2, output_token="combined"),
    "mantis-plus": dict(_BASE, family="mantis", arch="v1", hf_id="paris-noah/MantisPlus", params="8M",
                        layer=1, output_token="combined"),
    "mantis-v2": dict(_BASE, family="mantis-v2", arch="v2", hf_id="paris-noah/MantisV2", params="4M",
                      layer=2, output_token="combined"),
    "mantis-v2-cls": dict(_BASE, family="mantis-v2", arch="v2", hf_id="paris-noah/MantisV2", params="4M",
                          layer=-1, output_token="cls_token"),
}


class Encoder:
    def __init__(self, spec, device=None):
        import torch
        from mantis.architecture import MantisV1, MantisV2
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cls = MantisV1 if spec["arch"] == "v1" else MantisV2
        net = cls(device=self.device, return_transf_layer=spec["layer"], output_token=spec["output_token"])
        self.network = net.from_pretrained(spec["hf_id"]).eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

    def _resize(self, s, m):
        """Real points only, most recent `context` of them, linearly resized to seq_len."""
        import torch
        import torch.nn.functional as F
        x = np.asarray(s, np.float32)[np.asarray(m, bool)][-self.spec["context"]:]
        if len(x) == 0:
            x = np.zeros(1, np.float32)
        x = torch.from_numpy(np.ascontiguousarray(x)).view(1, 1, -1)
        return F.interpolate(x, size=self.spec["seq_len"], mode="linear", align_corners=False)[0]

    def encode(self, values, masks, times, batch_size=256):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(values), batch_size):
                x = torch.stack([self._resize(s, m) for s, m in zip(values[i:i + batch_size], masks[i:i + batch_size])])
                out.append(self.network(x.to(self.device)).float().cpu().numpy())
        return np.concatenate(out)
