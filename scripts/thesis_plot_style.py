from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


ROOT = Path(__file__).resolve().parent.parent

# Prefer full Computer Modern families when available, but fall back to
# Matplotlib's bundled cmr10 so thesis plots still render in a Computer Modern
# style on machines without a TeX font installation.
_FONT_CANDIDATES = (
    ROOT / "viewer" / "fonts" / "cmunrm.ttf",
    ROOT / "viewer" / "fonts" / "cmunbx.ttf",
    ROOT / "viewer" / "fonts" / "cmunti.ttf",
    ROOT / "viewer" / "fonts" / "cmunbi.ttf",
    ROOT / "viewer" / "fonts" / "lmroman10-regular.otf",
    ROOT / "viewer" / "fonts" / "lmroman10-bold.otf",
    ROOT / "viewer" / "fonts" / "lmroman10-italic.otf",
    ROOT / "viewer" / "fonts" / "lmroman10-bolditalic.otf",
    Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "fonts" / "opentype" / "public" / "lm" / "lmroman10-regular.otf",
    Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "fonts" / "opentype" / "public" / "lm" / "lmroman10-bold.otf",
    Path(r"C:\Program Files\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf"),
    Path(r"C:\Program Files\MiKTeX\fonts\opentype\public\lm\lmroman10-bold.otf"),
    Path(mpl.get_data_path()) / "fonts" / "ttf" / "cmr10.ttf",
)

COMPUTER_MODERN_SERIF_STACK = [
    "CMU Serif",
    "Computer Modern Roman",
    "Latin Modern Roman",
    "Computer Modern Serif",
    "cmr10",
    "DejaVu Serif",
]


def _register_available_fonts() -> None:
    seen: set[str] = set()
    for path in _FONT_CANDIDATES:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        fm.fontManager.addfont(resolved)
        seen.add(resolved)


def apply_thesis_style(*, font_size: float = 11, y_grid: bool = True) -> None:
    _register_available_fonts()
    plt.rcParams.update(
        {
            "font.family": ["serif"],
            "font.serif": COMPUTER_MODERN_SERIF_STACK,
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": y_grid,
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.7,
            "grid.linewidth": 1.0,
            "axes.grid.axis": "y",
        }
    )
