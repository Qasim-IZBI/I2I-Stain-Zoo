"""I2I-Stain-Zoo — unpaired image-to-image translation for virtual staining.

Six architectures (CycleGAN, UNIT, MUNIT, DCLGAN, UVCGAN, CycleDiffusion)
behind one training and inference interface.

Library entry points::

    from i2i_stain_zoo.models import CycleGAN, CycleGANConfig
    from i2i_stain_zoo.base_models import Encoder, Decoder, ResnetBottleneck
    from i2i_stain_zoo.datasets.unpaired_dataset import UnpairedDataset
    from i2i_stain_zoo.trainer.base_trainer import BaseTrainer

Command-line entry points live in ``i2i_stain_zoo.cli`` and are installed as
``i2i-train``, ``i2i-inference``, ``i2i-evaluate``, and so on.

Submodules are imported lazily so that ``import i2i_stain_zoo`` stays cheap and
does not pull in torch until something is actually used.
"""

__version__ = "1.0.0"

__all__ = ["base_models", "datasets", "models", "trainer", "utils", "cli"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
