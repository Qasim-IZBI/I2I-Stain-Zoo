"""I2I-Stain-Zoo — unpaired image-to-image translation for virtual staining.

Six architectures (CycleGAN, UNIT, MUNIT, DCLGAN, UVCGAN, CycleDiffusion)
behind one training and inference interface.

Library entry points::

    from i2i_stain_zoo.models import CycleGAN, CycleGANConfig
    from i2i_stain_zoo.base_models import Encoder, Decoder, ResnetBottleneck
    from i2i_stain_zoo.datasets.unpaired_dataset import UnpairedDataset
    from i2i_stain_zoo.trainer.base_trainer import BaseTrainer

Each command-line entry point is a module with a ``main()`` — ``train``,
``inference``, ``evaluation``, ``tile`` and so on — installed as ``i2i-train``,
``i2i-inference``, ``i2i-evaluate``. Their analysis functions are importable
directly::

    from i2i_stain_zoo.evaluation import compute_ssim_map, frechet_distance
    from i2i_stain_zoo.uncertainty_calibration import reliability_bins

Submodules are imported lazily so that ``import i2i_stain_zoo`` stays cheap and
does not pull in torch until something is actually used.
"""

__version__ = "1.0.0"

__all__ = [
    "base_models", "utils", "datasets", "models", "trainer",
    "train", "inference", "evaluation", "tile", "reconstruct",
    "uncertainty", "uncertainty_calibration",
    "aggregate_uncertainty", "aggregate_calibration",
    "compare_psr", "apply_he_mask", "fill_tissue_holes",
    "plot_combined_metrics", "plot_ranking_correlation", "plot_uncertainty_boxplot",
]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
