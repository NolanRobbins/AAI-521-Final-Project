"""DDColor architecture vendored from https://github.com/piddnad/DDColor (Apache-2.0).

Source commit: 2adb63f2656ac41cbdf7b894cddd94121a3faf13
Files: ddcolor/model.py and basicsr/archs/ddcolor_arch_utils/*. Only import paths were
changed (basicsr.archs.ddcolor_arch_utils -> .arch_utils). We vendor instead of
installing because upstream's setup.py compiles basicsr CUDA extensions at install time,
which is fragile on Windows and Colab, and we need only the network definition.

Kang et al., "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders",
ICCV 2023.
"""

from .model import DDColor

__all__ = ["DDColor"]
