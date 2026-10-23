"""Timestral (Mistral AI, 2026) encoders, via the `timestral` package (github.com/mistralai/timestral).

Timestral is a causal patch transformer (32 steps per patch) trained for quantile
forecasting with Decoupled Horizon Forecasting: context patches form a causal encoder
stream that never reads the forecast-query tokens. The embedding is the masked mean of
that encoder stream after the final norm, i.e. the states the forecast tokens attend to.

Weights: Modified MIT licence (attribution notice required in copies; no rights for
companies whose consolidated monthly revenue exceeds $20M). The inference code is
Apache-2.0 and not on PyPI: pip install "timestral @ git+https://github.com/mistralai/timestral.git".
"""
import numpy as np

_BASE = dict(family="timestral", license="Modified-MIT (Mistral AI; revenue cap)", extra="timestral",
             context=32768, patch=32, uses_time=False,
             install="pip install 'timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'")

MODELS = {
    "timestral-5m": dict(_BASE, hf_id="mistralai/Timestral-5M",
                         revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f", params="5M"),
    "timestral-28m": dict(_BASE, hf_id="mistralai/Timestral-28M",
                          revision="49b26409ac87b5935539a5d8ffe7f397b720db47", params="28M"),
    "timestral-123m": dict(_BASE, hf_id="mistralai/Timestral-123M",
                           revision="5a147287addda85f6acd8da3fb8858fe9af7e8f7", params="123M"),
    "timestral-209m": dict(_BASE, hf_id="mistralai/Timestral-209M",
                           revision="9874d252ab0aae16f7f7c999f98f98d6ad7d8365", params="209M"),
}


class Encoder:
    def __init__(self, spec, device=None):
        import torch
        from timestral import load_model
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(spec["hf_id"], device=self.device, revision=spec.get("revision"))

    def _encode_batch(self, x, observed):
        """(R, T) values and observation mask, T a multiple of the patch size -> (R, T / patch, d) encoder states.

        Mirrors TimestralModel._predict_context_only up to the transformer, with one forecast
        token appended (the masks need a horizon); the encoder rows never attend to it."""
        import torch
        from timestral.layers import TwoStreamMask
        from timestral.masks import (build_forecast_positions, build_forecast_time_mask,
                                     build_two_stream_variate_mask, compute_enc_key_validity,
                                     group_first_observed_patch)
        m, c = self.model, self.model.config
        rows, t = x.shape
        ll, h = t // c.patch_size, 1
        group_ids = torch.arange(rows, device=x.device)  # every curve is its own univariate series
        scaled, _, _ = m.scale(x, observed)
        x_enc = m.representation(scaled, observed)
        x_dec = m.representation.forecast_tokens(rows, h)
        obs_per_patch = observed.unflatten(-1, (ll, c.patch_size)).any(-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, h, key_valid, x.device)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
        first_obs = group_first_observed_patch(obs_per_patch, group_ids, rows)
        positions = build_forecast_positions(ll, h, first_obs)
        mask = TwoStreamMask(mask=time_mask, enc_len=ll) if c.split_context_horizon_attn else time_mask
        hid = m.transformer(torch.cat([x_enc, x_dec], 1), mask, variate_mask, positions)
        return hid[:, :ll]

    def encode(self, values, masks, times, batch_size=256):
        """Masked mean of encoder patch states. Curves are left-padded (as missing) to a common
        multiple of 32 steps; a patch counts if any of its points is real. As for Chronos, the
        values at mask=False positions are fed to the model; the mask only selects what is pooled."""
        import torch
        ctx_max, patch = self.spec["context"], self.spec["patch"]
        out = []
        with torch.inference_mode():
            for i in range(0, len(values), batch_size):
                s_b = [np.asarray(s, np.float32)[-ctx_max:] for s in values[i:i + batch_size]]
                m_b = [np.asarray(m, bool)[-ctx_max:] for m in masks[i:i + batch_size]]
                L = -(-max(len(s) for s in s_b) // patch) * patch
                x = torch.zeros((len(s_b), L))
                obs = torch.zeros((len(s_b), L), dtype=torch.bool)
                pool = torch.zeros((len(s_b), L), dtype=torch.bool)
                for j, (s, m) in enumerate(zip(s_b, m_b)):
                    fin = np.isfinite(s)
                    x[j, L - len(s):] = torch.from_numpy(np.where(fin, s, 0.0).astype(np.float32))
                    obs[j, L - len(s):] = torch.from_numpy(fin)
                    pool[j, L - len(s):] = torch.from_numpy(m & fin)
                emb = self._encode_batch(x.to(self.device), obs.to(self.device)).float().cpu()
                w = pool.view(len(s_b), L // patch, patch).any(-1).unsqueeze(-1).float()
                out.append(((emb * w).sum(1) / w.sum(1).clamp(min=1)).numpy())
        return np.concatenate(out)
