"""Kestrel — a sub-50M-parameter language model trained from scratch."""

__all__ = ["Kestrel", "KestrelConfig"]


def __getattr__(name: str):
    """Resolve the model lazily, so importing the data path does not import torch.

    prepare_data.py does `from kestrel.corpus import ...`, and importing any submodule executes
    this file. Eagerly re-exporting from kestrel.model dragged torch into the corpus build, where
    it is not merely wasteful but fatal: on a CPU container torch 2.13 loads its CUDA bindings, and
    those abort the interpreter at finalization when there is no GPU behind them -- SIGABRT after
    the corpus is already written, which reads as a failed build of a corpus that is actually fine.

    Nothing observable changes. Every caller in the tree already imports `kestrel.model` directly,
    and `from kestrel import Kestrel` still resolves through here. kestrel.variants does the same
    lazy import for the same reason.
    """
    if name in __all__:
        from kestrel import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
