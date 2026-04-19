"""AgriPipe: Excel agronomico grezzo → DataFrame pulito → tensor PyTorch.

Pipeline ML-Ops in 3 step:

1. :func:`agripipe.loader.load_raw` — carica Excel/CSV e valida lo schema.
2. :class:`agripipe.cleaner.AgriCleaner` — pulisce outlier/NaN/limiti fisici.
3. :class:`agripipe.tensorizer.Tensorizer` — trasforma in tensor PyTorch.
"""

from agripipe.cleaner import AgriCleaner, CleanerConfig
from agripipe.dataset import AgriDataset
from agripipe.loader import load_raw
from agripipe.report import generate_report
from agripipe.synth import SynthConfig, generate_dirty_excel
from agripipe.tensorizer import Tensorizer

__version__ = "0.2.0"

__all__ = [
    "AgriCleaner",
    "CleanerConfig",
    "AgriDataset",
    "Tensorizer",
    "load_raw",
    "generate_report",
    "generate_dirty_excel",
    "SynthConfig",
    "__version__",
]
