"""Shared chart styling so every figure in the report looks consistent."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Categorical slots in fixed order (CVD-validated reference palette)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREY = "#a3a29c"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "ytick.left": False,
            "text.color": INK,
            "legend.frameon": False,
        }
    )


def inr(x, _pos=None) -> str:
    """Format a rupee value: ₹42K, ₹4,529."""
    sign = "−" if x < 0 else ""
    x = abs(x)
    if x >= 10_000:
        return f"{sign}₹{x/1000:,.0f}K"
    return f"{sign}₹{x:,.0f}"


def inr_axis(ax, axis: str = "y") -> None:
    fmt = mticker.FuncFormatter(inr)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = mticker.PercentFormatter(xmax=1, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def subtitle(ax, text: str) -> None:
    """Grey one-line subtitle under the bold title."""
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, facecolor="white")
    return path
