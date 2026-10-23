from . import tasks, transforms
from .compare import Results, compare
from .embedders import Embedder, available_models, prefetch
from .features import Catalog, HandCrafted
from .lightcurves import LightCurve, LightCurves

__version__ = "0.1.1"
__all__ = ["LightCurve", "LightCurves", "transforms", "tasks", "Embedder", "available_models", "prefetch",
           "HandCrafted", "Catalog", "compare", "Results"]
