"""The six unpaired translation architectures."""

from i2i_stain_zoo.models.cyclegan import CycleGAN, CycleGANConfig
from i2i_stain_zoo.models.unit import UNIT, UNITConfig
from i2i_stain_zoo.models.munit import MUNIT, MUNITConfig
from i2i_stain_zoo.models.dclgan import DCLGAN, DCLGANConfig
from i2i_stain_zoo.models.uvcgan import UVCGAN, UVCGANConfig
from i2i_stain_zoo.models.cyclediffusion import CycleDiffusion, CycleDiffusionConfig

__all__ = [
    "CycleGAN", "CycleGANConfig",
    "UNIT", "UNITConfig",
    "MUNIT", "MUNITConfig",
    "DCLGAN", "DCLGANConfig",
    "UVCGAN", "UVCGANConfig",
    "CycleDiffusion", "CycleDiffusionConfig",
]
