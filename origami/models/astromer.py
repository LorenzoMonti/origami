"""Astromer 1 (Donoso-Oliva et al. 2023) and Astromer 2 (Donoso-Oliva et al. 2026) encoders.

Astromer is a TensorFlow transformer that takes (time, magnitude) pairs, i.e. irregular
sampling, with a sinusoidal positional encoding of the observation time. The official
TensorFlow checkpoints (Zenodo 18207945, CC-BY-4.0; Zenodo 16410988, GPL-3.0-or-later for
the ZTF retraining) are loaded through their ONNX exports by the light-curve project
(github.com/light-curve/prep-models, tf2onnx of the encoder sublayer), run with
onnxruntime: no TensorFlow needed. Revisions are pinned.

Input protocol of the published weights (prep-models README, upstream
src/data/preprocessing.py): fixed windows of 200 observations; magnitudes and times
centred on their per-window mean (no scaling); zero right-padding; mask 1 = real.
"""
import numpy as np

_WINDOW = 200
_COMMON = dict(extra="astromer", context=_WINDOW, uses_time=True, dim=256, heads=4, head_dim=64)

MODELS = {
    "astromer1-macho": dict(
        _COMMON, family="astromer1", hf_id="light-curve/astromer1", onnx_file="astromer1.onnx",
        revision="9ee820a1fdb3f55b2a221163d9c74ac1809677c8", layers=2, params="0.66M",
        license="CC-BY-4.0 (weights, Zenodo 18207945); MIT (ONNX export)",
        source="https://zenodo.org/records/18207945 pt_macho_v1_2021", trained_on="MACHO R"),
    "astromer2-macho": dict(
        _COMMON, family="astromer2", hf_id="light-curve/astromer2", onnx_file="astromer2.onnx",
        revision="17752a44aaaa1cd4a014be53e463ee06b61e68f0", layers=6, params="4.0M",
        license="CC-BY-4.0 (weights, Zenodo 18207945); MIT (ONNX export)",
        source="https://zenodo.org/records/18207945 pt_macho_v2_2025", trained_on="MACHO V+R, 1.5M curves"),
    "astromer1-ztfdr20": dict(
        _COMMON, family="astromer1", hf_id="light-curve/astromer1-ztfdr20", onnx_file="astromer1_ztfdr20.onnx",
        revision="d2855ff0aada7d8170dc7433afaebfa92ea1a34a", layers=2, params="0.66M",
        license="GPL-3.0-or-later", source="https://zenodo.org/records/16410988 ANN_clf (encoder only)",
        trained_on="ZTF DR20 g (Nakoneczny et al. 2025, QSO/galaxy/star classifier)"),
}


class Encoder:
    def __init__(self, spec, device=None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        self.spec = spec
        path = hf_hub_download(spec["hf_id"], spec["onnx_file"], revision=spec.get("revision"))
        providers = ["CPUExecutionProvider"]
        if device is not None and str(device).startswith("cuda") \
                and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(path, providers=providers)

    @staticmethod
    def window(values, masks, times, W=_WINDOW):
        """Model inputs for one curve: real points only, the most recent W of them, magnitudes
        and times centred on the window mean (Astromer's `standardize`), zero right-padding."""
        v, t = np.asarray(values, np.float64), np.asarray(times, np.float64)
        keep = np.asarray(masks, bool) & np.isfinite(v) & np.isfinite(t)
        v, t = v[keep][-W:], t[keep][-W:]
        inp, tim, msk = np.zeros(W, np.float32), np.zeros(W, np.float32), np.zeros(W, np.float32)
        n = len(v)
        if n:
            inp[:n], tim[:n], msk[:n] = v - v.mean(), t - t.mean(), 1.0
        return inp, tim, msk

    def encode(self, values, masks, times, batch_size=256, scales=None):
        """Masked mean over real positions of the per-observation encoder outputs (no special
        tokens in Astromer). Curves with more than 200 real points: last 200 only.
        scales: per-curve std removed by origami's z-normalisation; Astromer was trained on
        mean-subtracted, unscaled magnitudes, so values are multiplied back by it."""
        if scales is not None:
            values = [np.asarray(v, np.float64) * s for v, s in zip(values, scales)]
        out = []
        for i in range(0, len(values), batch_size):
            w = [self.window(v, m, t) for v, m, t in zip(values[i:i + batch_size], masks[i:i + batch_size],
                                                          times[i:i + batch_size])]
            inp, tim, msk = (np.stack(x)[..., None] for x in zip(*w))
            seq, = self.session.run(["sequence"], {"input": inp, "times": tim, "mask_in": msk})
            out.append((seq * msk).sum(1) / np.maximum(msk.sum(1), 1.0))
        return np.concatenate(out).astype(np.float32)
