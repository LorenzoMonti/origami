"""Moirai 1.1 (Woo et al. 2024) and Moirai 2.0 (Liu et al. 2025) encoders, via the uni2ts package.

Both are patch-based transformer encoders over regularly sampled series: timestamps are
not used (positions are integer patch indices with rotary embeddings). Weights are
released under CC-BY-NC-4.0 (non-commercial use only).
"""
import numpy as np

_V1 = dict(family="moirai", license="CC-BY-NC-4.0", extra="moirai", uses_time=False, patch=16,
           max_tokens=512, context=512 * 16)
_V2 = dict(family="moirai-2", license="CC-BY-NC-4.0", extra="moirai", uses_time=False, patch=16,
           max_tokens=512, context=512 * 16)

TOKEN_BUDGET = 16384  # max batch x patches per forward pass (memory guard)

MODELS = {
    "moirai-1.1-small": dict(_V1, hf_id="Salesforce/moirai-1.1-R-small", params="14M"),
    "moirai-1.1-base": dict(_V1, hf_id="Salesforce/moirai-1.1-R-base", params="91M"),
    "moirai-2.0-small": dict(_V2, hf_id="Salesforce/moirai-2.0-R-small", params="11M"),
}


class Encoder:
    def __init__(self, spec, device=None):
        import torch
        self.spec = spec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if spec["family"] == "moirai-2":
            from uni2ts.model.moirai2 import Moirai2Module as Module
        else:
            from uni2ts.model.moirai import MoiraiModule as Module
        self.module = Module.from_pretrained(spec["hf_id"]).to(self.device).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if spec["family"] == "moirai-2" and self.module.patch_size != spec["patch"]:
            raise ValueError(f"{spec['hf_id']} has patch size {self.module.patch_size}, spec says {spec['patch']}")
        if spec["family"] == "moirai" and spec["patch"] not in self.module.patch_sizes:
            raise ValueError(f"patch {spec['patch']} not in {self.module.patch_sizes}")

    def _hidden(self, target, observed, sample_id, time_id):
        """Final-layer encoder states (after the last RMSNorm), reproducing the module's
        forward pass up to the output head. No prediction tokens: every token is context."""
        import torch
        from uni2ts.common.torch_util import packed_attention_mask, packed_causal_attention_mask
        m = self.module
        variate_id = torch.zeros_like(sample_id)
        loc, scale = m.scaler(target, observed, sample_id, variate_id)
        x = (target - loc) / scale
        if self.spec["family"] == "moirai-2":
            reprs = m.in_proj(torch.cat([x, observed.float()], -1))
            attn = packed_causal_attention_mask(sample_id, time_id)
        else:
            # MultiInSizeLinear expects inputs zero-padded to the largest patch size
            x = torch.nn.functional.pad(x, (0, max(m.patch_sizes) - x.shape[-1]))
            patch_size = torch.full_like(time_id, self.spec["patch"])
            reprs = m.in_proj(x, patch_size)
            attn = packed_attention_mask(sample_id)
        return m.encoder(reprs, attn, time_id=time_id, var_id=variate_id)

    def encode(self, values, masks, times, batch_size=256):
        """Masked mean of final-layer patch states. Each curve (last `context` points) is
        left-padded to a multiple of the patch size, then whole padding patches are added on
        the left up to the batch's longest curve. Padding patches get sample_id 0, which the
        attention mask isolates from the real ones (sample_id 1) and the scaler ignores; points
        that are padding or masked are passed as unobserved with value 0. A patch is pooled
        if any of its points is real. There are no special tokens. Batches are split further
        so that a forward pass holds at most TOKEN_BUDGET tokens."""
        patch, ctx_max = self.spec["patch"], self.spec["context"]
        s_all = [np.asarray(s, np.float32)[-ctx_max:] for s in values]
        m_all = [np.asarray(m, bool)[-ctx_max:] for m in masks]
        out = []
        for i in range(0, len(s_all), batch_size):
            T = max(-(-len(s) // patch) for s in s_all[i:i + batch_size])
            sub = max(1, min(batch_size, TOKEN_BUDGET // T))
            for k in range(i, min(i + batch_size, len(s_all)), sub):
                out.append(self._encode_batch(s_all[k:min(k + sub, i + batch_size)], m_all[k:min(k + sub, i + batch_size)]))
        return np.concatenate(out)

    def _encode_batch(self, s_b, m_b):
        import torch
        patch = self.spec["patch"]
        n_tok = [-(-len(s) // patch) for s in s_b]
        T, B = max(n_tok), len(s_b)
        tgt = torch.zeros((B, T * patch))
        obs = torch.zeros((B, T * patch), dtype=torch.bool)
        sample_id = torch.zeros((B, T), dtype=torch.long)
        time_id = torch.zeros((B, T), dtype=torch.long)
        for j, (s, m) in enumerate(zip(s_b, m_b)):
            real = m & np.isfinite(s)
            tgt[j, T * patch - len(s):] = torch.from_numpy(np.where(real, s, 0.0).astype(np.float32))
            obs[j, T * patch - len(s):] = torch.from_numpy(real)
            sample_id[j, T - n_tok[j]:] = 1
            time_id[j, T - n_tok[j]:] = torch.arange(n_tok[j])
        tgt, obs = tgt.view(B, T, patch).to(self.device), obs.view(B, T, patch).to(self.device)
        sample_id, time_id = sample_id.to(self.device), time_id.to(self.device)
        with torch.no_grad():
            h = self._hidden(tgt, obs, sample_id, time_id).float()
        w = (obs.any(-1) & (sample_id == 1)).unsqueeze(-1).float()
        return ((h * w).sum(1) / w.sum(1).clamp(min=1)).cpu().numpy()
