"""
ESMFold backend: inference and trace export via OpenFold model loading.

Uses fair-esm to download the ESMFold checkpoint, patches IPA keys for
VizFold compatibility, then runs inference through OpenFold's AlphaFold
class to leverage the existing ATTENTION_METADATA capture pipeline.
"""


# Lazy import so schema/ can be used without requiring torch/fair-esm
def __getattr__(name):
    if name == "ESMFoldRunner":
        from vizfold.backends.esmfold.inference import ESMFoldRunner
        return ESMFoldRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ESMFoldRunner"]
