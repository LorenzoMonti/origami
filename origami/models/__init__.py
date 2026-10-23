"""
Model plugins. Each module in this package declares

  MODELS = {name: spec}   spec: dict with at least family, hf_id (or source),
                          license, extra (pip extra that installs its
                          dependencies), context (max input length)
  class Encoder:
      def __init__(self, spec, device=None): ...   # loads weights; heavy imports here, not at module level
      def encode(self, values, masks, times, batch_size=256) -> np.ndarray  # (n_curves, dim)
      # optional extra parameter `scales`: per-curve std (magnitudes) removed by z-normalisation;
      # declare it only if the model needs magnitude units (values * scale)

`values`, `masks` and `times` are the lists returned by an input transform
(see origami.transforms). The encoder pools its own per-token or per-patch
states over the positions that come from real data. Modules are discovered
automatically: adding a model means adding one file here.
"""
import importlib
import pkgutil

_REGISTRY = None


def registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {}
        for info in pkgutil.iter_modules(__path__):
            module = importlib.import_module(f"{__name__}.{info.name}")
            for name, spec in getattr(module, "MODELS", {}).items():
                if name in _REGISTRY:
                    raise RuntimeError(f"model {name!r} declared twice ({_REGISTRY[name]['module']}, {info.name})")
                _REGISTRY[name] = dict(spec, module=info.name)
    return _REGISTRY


def get_encoder(name, device=None):
    spec = registry()[name]
    module = importlib.import_module(f"{__name__}.{spec['module']}")
    try:
        return module.Encoder(spec, device=device)
    except ImportError as e:
        hint = f" (Python >= {spec['requires_python']})" if spec.get("requires_python") else ""
        how = spec.get("install") or f"pip install origami-astro[{spec['extra']}]"
        raise ImportError(f"{name} needs: {how}{hint}") from e
