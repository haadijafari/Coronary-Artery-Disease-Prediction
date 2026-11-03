"""Charts for the analysis notebook and the README, in one consistent style.

Colour carries one meaning everywhere: blue is "no disease" (or pushes the
risk down) and orange is "disease" (or pushes the risk up). The two-colour
palette was checked for colour-vision deficiency (worst CVD ΔE 24.7, normal
vision ΔE 33.6 on the chart surface). Everything not carrying data is drawn in
recessive greys, and every figure leads with a title that states its finding.
"""

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure, SubFigure
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from heart_disease.config import FIGURES_DIR
from heart_disease.interpretation import disease_rate_by_category
from heart_disease.schema import CLASS_LABELS, FEATURES_BY_NAME, TARGET

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b5b3ab"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
CLASS_COLORS = {0: BLUE, 1: ORANGE}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("blue_orange", [BLUE, "#f0efec", ORANGE])

_FONTS = ("Lato", "DejaVu Sans")  # both include U+2212, the minus sign used on axes
_TITLE_SIZE = 13
_SUBTITLE_SIZE = 9.5


def set_theme() -> None:
    """Apply the project style to matplotlib."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next((font for font in _FONTS if font in available), "sans-serif")
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "figure.constrained_layout.use": True,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.titlepad": 8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9.5,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0,
        }
    )


def save_figure(fig: Figure, name: str, directory: str | Path = FIGURES_DIR) -> Path:
    """Write ``fig`` as a PNG named ``name`` and return its path."""
    path = Path(directory) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    return path


def _figure(
    title: str,
    subtitle: str | None,
    size: tuple[float, float],
    legend: list | None = None,
) -> tuple[Figure, SubFigure]:
    """A figure split into a header (title, subtitle, legend) and a body for the axes.

    ``size`` is the body size in inches; the header height is added on top, so
    neither the title nor the legend can collide with the plots.
    """
    width, body_height = size
    lines = textwrap.wrap(subtitle, int(width * 14)) if subtitle else []
    title_pt, line_pt, legend_pt, pad_pt = _TITLE_SIZE * 1.35, _SUBTITLE_SIZE * 1.4, 20, 8
    header_pt = pad_pt + title_pt + len(lines) * line_pt + (legend_pt if legend else 0) + pad_pt
    header_height = header_pt / 72

    fig = plt.figure(figsize=(width, body_height + header_height))
    header, body = fig.subfigures(2, 1, height_ratios=[header_height, body_height])
    header.text(
        0.005, 1 - pad_pt / header_pt, title, ha="left", va="top", fontsize=_TITLE_SIZE,
        fontweight="semibold", color=INK,
    )  # fmt: skip
    if lines:
        header.text(
            0.005, 1 - (pad_pt + title_pt) / header_pt, "\n".join(lines), ha="left", va="top",
            fontsize=_SUBTITLE_SIZE, color=INK_SECONDARY, linespacing=1.4,
        )  # fmt: skip
    if legend:
        header.legend(
            handles=legend, loc="lower left", bbox_to_anchor=(0, 0), ncols=len(legend),
            handletextpad=0.4, columnspacing=1.4, borderaxespad=0, borderpad=0.2,
        )  # fmt: skip
    return fig, body


def _class_handles() -> list:
    return [
        mpl.lines.Line2D([], [], color=CLASS_COLORS[k], marker="o", lw=0, label=CLASS_LABELS[k])
        for k in (0, 1)
    ]


def _label(name: str) -> str:
    feature = FEATURES_BY_NAME[name]
    return f"{feature.label} ({feature.unit})" if feature.unit else feature.label


def _x_grid_only(ax) -> None:
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


# --- Data audit ---------------------------------------------------------------------------


def plot_iqr_audit(dataset: pd.DataFrame, columns: Sequence[str]) -> Figure:
    """What a 1.5 x IQR outlier filter would delete, shown against each patient's diagnosis."""
    flagged_by_column, deleted = {}, pd.Series(False, index=dataset.index)
    for column in columns:
        q1, q3 = dataset[column].quantile([0.25, 0.75])
        fences = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        flagged = (dataset[column] < fences[0]) | (dataset[column] > fences[1])
        flagged_by_column[column] = (fences, flagged)
        deleted |= flagged
    n_cases = int(dataset.loc[deleted, TARGET].sum())

    fig, body = _figure(
        f"An IQR filter would delete {int(deleted.sum())} real patients, {n_cases} with disease",
        "Ringed points fall outside the 1.5 × IQR fences. They are high but clinically "
        "plausible readings, and deleting them would hide exactly the high-risk patients.",
        (8.5, 1.9 * len(columns)),
        legend=_class_handles(),
    )
    axes = np.atleast_1d(body.subplots(len(columns), 1))
    rng = np.random.default_rng(0)
    jitter = pd.Series(rng.uniform(-0.18, 0.18, len(dataset)), index=dataset.index)
    for ax, column in zip(axes, columns, strict=True):
        (low, high), flagged = flagged_by_column[column]
        values = dataset[column]
        for cls in (0, 1):
            for is_flagged in (False, True):
                mask = (dataset[TARGET] == cls) & (flagged == is_flagged)
                ax.scatter(
                    values[mask], cls + jitter[mask], s=48 if is_flagged else 14,
                    color=CLASS_COLORS[cls], alpha=1 if is_flagged else 0.35,
                    edgecolors=INK if is_flagged else "none", linewidths=1.2 if is_flagged else 0,
                    zorder=3 if is_flagged else 2,
                )  # fmt: skip
        for fence in (low, high):
            if values.min() <= fence <= values.max():
                ax.axvline(fence, color=INK_MUTED, lw=1, zorder=1)
                ax.text(fence, 1.5, " IQR fence", color=INK_MUTED, fontsize=8.5, va="center")
        ax.set_yticks([0, 1], [CLASS_LABELS[0], CLASS_LABELS[1]])
        ax.set_ylim(-0.45, 1.65)
        ax.set_xlabel(_label(column))
        _x_grid_only(ax)
    return fig


# --- Exploration --------------------------------------------------------------------------


def plot_numeric_distributions(
    dataset: pd.DataFrame, columns: Sequence[str], title: str = "Distributions by diagnosis"
) -> Figure:
    """Density of each numeric feature by diagnosis, as small multiples."""
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, body = _figure(
        title,
        "Kernel density per group; small ticks on the baseline mark each group's median.",
        (10, 2.5 * n_rows),
        legend=_class_handles(),
    )
    axes = body.subplots(n_rows, n_cols).ravel()
    for ax, column in zip(axes, columns, strict=False):
        for cls in (0, 1):
            values = dataset.loc[dataset[TARGET] == cls, column].dropna()
            sns.kdeplot(
                values, ax=ax, color=CLASS_COLORS[cls], fill=True, alpha=0.12, cut=0,
                linewidth=2,
            )  # fmt: skip
            ax.axvline(values.median(), color=CLASS_COLORS[cls], lw=1.5, ymax=0.08)
        ax.set_xlabel(_label(column))
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    for ax in axes[len(columns) :]:
        ax.set_visible(False)
    return fig


def _balanced_columns(sizes: Sequence[int]) -> int:
    """Index at which to split panels into two columns of similar total height."""
    total, running = sum(sizes), 0
    for i, size in enumerate(sizes):
        running += size
        if running >= total / 2:
            return i + 1
    return len(sizes)


def plot_disease_rates(
    dataset: pd.DataFrame, columns: Sequence[str], title: str = "Disease rate by category"
) -> Figure:
    """Share of patients with disease per category, with 95% Wilson intervals."""
    rates = {column: disease_rate_by_category(dataset, column) for column in columns}
    sizes = [len(rates[c]) + 1.6 for c in columns]  # rows plus room for the panel title
    split_at = _balanced_columns(sizes)
    groups = [list(columns[:split_at]), list(columns[split_at:])]
    prevalence = dataset[TARGET].mean()

    fig, body = _figure(
        title,
        f"Dot = share with disease, bar = 95% Wilson interval, vertical line = overall rate "
        f"({prevalence:.0%}). Wide bars flag small, uncertain groups.",
        (10.5, 0.36 * max(sum(sizes[:split_at]), sum(sizes[split_at:]))),
    )
    for subfigure, group in zip(body.subfigures(1, 2, wspace=0.04), groups, strict=True):
        group_sizes = [len(rates[c]) + 1.6 for c in group]
        axes = subfigure.subplots(len(group), 1, height_ratios=group_sizes, sharex=True)
        for ax, column in zip(np.atleast_1d(axes), group, strict=True):
            table = rates[column]
            y = np.arange(len(table))[::-1]
            labels = [
                f"{int(c) if isinstance(c, float) else c}  (n={n})"
                for c, n in zip(table["category"], table["n"], strict=True)
            ]
            ax.axvline(prevalence, color=AXIS, lw=1, zorder=1)
            ax.hlines(y, table["lower"], table["upper"], color=ORANGE, alpha=0.35, lw=4)
            ax.scatter(table["rate"], y, color=ORANGE, s=44, zorder=3, edgecolors=SURFACE, lw=1)
            ax.set_yticks(y, labels)
            ax.set_ylim(-0.6, len(table) - 0.4)
            ax.set_title(FEATURES_BY_NAME[column].label)
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
            _x_grid_only(ax)
    return fig


def plot_correlation(
    dataset: pd.DataFrame, columns: Sequence[str], title: str = "Correlation between measurements"
) -> Figure:
    """Spearman rank correlation between numeric features and the target."""
    names = [*columns, TARGET]
    corr = dataset[names].corr(method="spearman")
    labels = [FEATURES_BY_NAME[c].label if c != TARGET else "Disease" for c in names]
    corr.index, corr.columns = labels, labels
    lower = corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(lower, dtype=bool), k=1)

    fig, body = _figure(
        title,
        "Spearman rank correlation, which also suits the 0–3 vessel count. "
        "The bottom row is each measurement's association with disease.",
        (7.2, 5.4),
    )
    ax = body.subplots()
    sns.heatmap(
        lower, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, annot=True, fmt=".2f",
        annot_kws={"fontsize": 9}, linewidths=2, linecolor=SURFACE, square=True,
        cbar_kws={"shrink": 0.7, "label": "Spearman ρ"}, ax=ax,
    )  # fmt: skip
    ax.grid(False)
    ax.tick_params(axis="x", rotation=35)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    return fig


def plot_age_vs_heart_rate(
    dataset: pd.DataFrame, title: str = "Maximum heart rate against age"
) -> Figure:
    """Maximum heart rate against age, coloured by diagnosis, with a linear trend per group."""
    fig, body = _figure(
        title,
        "Each dot is a patient; lines are least-squares trends within each group.",
        (8, 4.6),
        legend=_class_handles(),
    )
    ax = body.subplots()
    for cls in (0, 1):
        subset = dataset[dataset[TARGET] == cls]
        ax.scatter(
            subset["age"], subset["max_heart_rate"], s=30, color=CLASS_COLORS[cls], alpha=0.75,
            edgecolors=SURFACE, linewidths=0.8,
        )  # fmt: skip
        slope, intercept = np.polyfit(subset["age"], subset["max_heart_rate"], 1)
        xs = np.array([subset["age"].min(), subset["age"].max()])
        ax.plot(xs, intercept + slope * xs, color=CLASS_COLORS[cls], lw=2.2)
    ax.set_xlabel(_label("age"))
    ax.set_ylabel(_label("max_heart_rate"))
    return fig


# --- Modelling ----------------------------------------------------------------------------


def plot_model_comparison(scores: pd.DataFrame, selected: str, metric: str = "roc_auc") -> Figure:
    """Nested cross-validation scores per model: every outer fold, mean and ±1 SD."""
    data = scores[scores["metric"] == metric]
    order = data.groupby("model", sort=False)["value"].mean().sort_values().index
    fig, body = _figure(
        f"Three models tie; the simplest, {selected.lower()}, is kept",
        "Nested cross-validation on the training set. Small dots = outer folds, large dot = "
        "mean, bar = ±1 SD. Rule: simplest model within one standard error of the best.",
        (8.5, 0.6 * len(order) + 0.7),
    )
    ax = body.subplots()
    rng = np.random.default_rng(0)
    for y, model in enumerate(order):
        values = data.loc[data["model"] == model, "value"]
        chosen = model == selected
        color = BLUE if chosen else NEUTRAL
        ax.scatter(
            values, y + rng.uniform(-0.16, 0.16, len(values)), s=12, color=color, alpha=0.45, lw=0
        )
        mean, sd = values.mean(), values.std()
        ax.hlines(y, mean - sd, mean + sd, color=color, lw=2.5)
        ax.scatter(mean, y, s=70, color=color, edgecolors=SURFACE, lw=1.5, zorder=3)
        ax.annotate(
            f"{mean:.3f}", (max(mean + sd, values.max()), y), xytext=(8, 0),
            textcoords="offset points", va="center", fontsize=9.5,
            color=INK if chosen else INK_SECONDARY, fontweight="semibold" if chosen else "normal",
        )  # fmt: skip
    ax.set_yticks(range(len(order)), order)
    for tick in ax.get_yticklabels():
        if tick.get_text() == selected:
            tick.set_color(INK)
            tick.set_fontweight("semibold")
    ax.set_xlabel("ROC AUC on the outer folds (higher is better)")
    ax.set_xlim(0.4, 1.04)
    _x_grid_only(ax)
    return fig


def plot_threshold_tradeoff(y_true, proba, threshold: float, target_recall: float) -> Figure:
    """Sensitivity and specificity across thresholds, with the chosen operating point."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    grid = np.linspace(0, 1, 401)
    sensitivity = [np.mean(proba[y_true == 1] >= t) for t in grid]
    specificity = [np.mean(proba[y_true == 0] < t) for t in grid]
    sens_at = np.mean(proba[y_true == 1] >= threshold)
    spec_at = np.mean(proba[y_true == 0] < threshold)

    fig, body = _figure(
        f"Catching {target_recall:.0%} of cases means accepting more false alarms",
        "Out-of-fold predictions on the training set. The chosen threshold is the highest one "
        f"that still flags at least {target_recall:.0%} of disease cases.",
        (8, 4.2),
        legend=[
            mpl.lines.Line2D([], [], color=ORANGE, label="Sensitivity: disease cases caught"),
            mpl.lines.Line2D([], [], color=BLUE, label="Specificity: healthy patients cleared"),
        ],
    )
    ax = body.subplots()
    ax.axhline(target_recall, color=AXIS, lw=1)
    ax.axvline(threshold, color=INK_SECONDARY, lw=1)
    ax.plot(grid, sensitivity, color=ORANGE, drawstyle="steps-post")
    ax.plot(grid, specificity, color=BLUE, drawstyle="steps-post")
    ax.scatter([threshold] * 2, [sens_at, spec_at], color=[ORANGE, BLUE], s=50, zorder=3,
               edgecolors=SURFACE, lw=1.5)  # fmt: skip
    ax.annotate(
        f"Threshold {threshold:.2f}\nsensitivity {sens_at:.0%}\nspecificity {spec_at:.0%}",
        (threshold, 0.3), xytext=(10, 0), textcoords="offset points", fontsize=9,
        color=INK_SECONDARY, va="center", linespacing=1.4,
    )  # fmt: skip
    ax.annotate(
        f"target {target_recall:.0%}", (1, target_recall), xytext=(0, -12),
        textcoords="offset points", ha="right", fontsize=8.5, color=INK_MUTED,
    )  # fmt: skip
    ax.set_xlabel("Decision threshold on the predicted probability of disease")
    ax.set_ylabel("Share of patients")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    return fig


def plot_roc(y_true, proba, operating_points: Mapping[str, float], ax=None):
    """ROC curve with labelled operating points."""
    ax = ax or plt.subplots(figsize=(4.5, 4.2))[1]
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.plot(fpr, tpr, color=BLUE, lw=2, drawstyle="steps-post")
    for name, threshold in operating_points.items():
        sens = np.mean(proba[y_true == 1] >= threshold)
        fall_out = np.mean(proba[y_true == 0] >= threshold)
        ax.scatter(fall_out, sens, s=55, color=ORANGE, edgecolors=SURFACE, lw=1.5, zorder=3)
        ax.annotate(
            f"{name}\n{sens:.0%} sensitivity\n{1 - fall_out:.0%} specificity", (fall_out, sens),
            xytext=(10, -6), textcoords="offset points", fontsize=8.5, color=INK_SECONDARY,
            va="top", linespacing=1.3,
        )  # fmt: skip
    ax.set_title(f"ROC curve, AUC {roc_auc_score(y_true, proba):.3f}")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="x", visible=True)
    return ax


def plot_confusion(y_true, proba, threshold: float, ax=None):
    """Confusion matrix at ``threshold`` with counts and row percentages."""
    ax = ax or plt.subplots(figsize=(4.2, 3.8))[1]
    predicted = (np.asarray(proba) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for (i, j), count in np.ndenumerate(matrix):
        color = "white" if shares[i, j] > 0.55 else INK
        ax.text(j, i - 0.07, f"{count}", ha="center", va="center", fontsize=16, color=color,
                fontweight="semibold")  # fmt: skip
        ax.text(j, i + 0.2, f"{shares[i, j]:.0%} of row", ha="center", va="center",
                fontsize=8.5, color=color)  # fmt: skip
    ax.set_xticks([0, 1], [CLASS_LABELS[0], CLASS_LABELS[1]])
    ax.set_yticks([0, 1], [CLASS_LABELS[0], CLASS_LABELS[1]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix at threshold {threshold:.2f}")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_calibration(y_true, proba, n_bins: int = 5, ax=None):
    """Reliability diagram with quantile bins."""
    ax = ax or plt.subplots(figsize=(4.5, 4.2))[1]
    observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.plot(predicted, observed, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
    ax.set_title(f"Calibration in {n_bins} equal-size bins")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed share with disease")
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="x", visible=True)
    return ax


def plot_test_performance(y_true, proba, threshold: float) -> Figure:
    """ROC, confusion matrix and calibration on the held-out test set, side by side."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    caught, cases = int(np.sum((proba >= threshold) & (y_true == 1))), int(np.sum(y_true == 1))
    fig, body = _figure(
        f"On unseen patients the model catches {caught} of {cases} disease cases",
        f"{len(y_true)} patients the model never saw during selection, tuning or "
        "thresholding. Calibration bins are wide because the set is small.",
        (13.5, 4.4),
    )
    axes = body.subplots(1, 3)
    operating_points = {f"Chosen ({threshold:.2f})": threshold, "Default (0.50)": 0.5}
    plot_roc(y_true, proba, operating_points, axes[0])
    plot_confusion(y_true, proba, threshold, axes[1])
    plot_calibration(y_true, proba, ax=axes[2])
    return fig


# --- Interpretation -----------------------------------------------------------------------


def _term_label(feature: str, level: str | None, scale: float | None) -> str:
    spec = FEATURES_BY_NAME[feature]
    if pd.isna(level):
        unit = f" {spec.unit}" if spec.unit else ""
        return f"{spec.label}, +{scale:.2g}{unit}" if scale else spec.label
    return f"{spec.label}: {level} vs {spec.categories[0].lower()}"


def plot_odds_ratios(
    table: pd.DataFrame, scales: Mapping[str, float], title: str = "What the model has learned"
) -> Figure:
    """Forest plot of odds ratios with bootstrap intervals, on a log scale."""
    table = table.sort_values("odds_ratio").reset_index(drop=True)
    fig, body = _figure(
        title,
        "Odds ratios with 95% bootstrap intervals, log scale. Numeric terms are per one standard "
        "deviation. Grey intervals cross 1. L2 regularisation shrinks every ratio towards 1.",
        (9, 0.34 * len(table) + 0.8),
        legend=[
            mpl.lines.Line2D([], [], color=BLUE, marker="o", lw=0, label="Lowers the odds"),
            mpl.lines.Line2D([], [], color=ORANGE, marker="o", lw=0, label="Raises the odds"),
        ],
    )
    ax = body.subplots()
    ax.axvline(1, color=AXIS, lw=1)
    for y, row in table.iterrows():
        clear = row["lower"] > 1 or row["upper"] < 1
        color = (ORANGE if row["odds_ratio"] > 1 else BLUE) if clear else NEUTRAL
        ax.hlines(y, row["lower"], row["upper"], color=color, lw=2.4, alpha=0.55)
        ax.scatter(row["odds_ratio"], y, s=42, color=color, edgecolors=SURFACE, lw=1, zorder=3)
    labels = [
        _term_label(f, lvl, scales.get(f))
        for f, lvl in zip(table["feature"], table["level"], strict=True)
    ]
    ax.set_yticks(range(len(table)), labels)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(table["lower"].min() * 0.8, table["upper"].max() * 1.25)
    ax.set_ylim(-0.7, len(table) - 0.3)
    ax.set_xlabel("Odds ratio for disease")
    _x_grid_only(ax)
    return fig


def plot_patient_explanation(
    patient: pd.Series, contributions: pd.Series, baseline_log_odds: float, top: int = 8
) -> Figure:
    """Each feature's push on one patient's log-odds, relative to an average patient."""
    order = contributions.abs().sort_values().index[-top:]
    values = contributions[order]
    proba = 1 / (1 + np.exp(-(baseline_log_odds + contributions.sum())))
    fig, body = _figure(
        f"Predicted probability of disease: {proba:.0%}",
        f"The {top} largest contributions to this patient's log-odds, relative to an average "
        "patient. Contributions add up exactly to the model's prediction.",
        (8, 0.4 * len(order) + 0.6),
        legend=[
            mpl.patches.Patch(color=BLUE, label="Towards no disease"),
            mpl.patches.Patch(color=ORANGE, label="Towards disease"),
        ],
    )
    ax = body.subplots()
    ax.barh(range(len(order)), values, color=[ORANGE if v > 0 else BLUE for v in values],
            height=0.6)  # fmt: skip
    ax.axvline(0, color=AXIS, lw=1)
    labels = []
    for name in order:
        value = patient[name]
        shown = f"{value:g}" if isinstance(value, int | float | np.number) else value
        labels.append(f"{FEATURES_BY_NAME[name].label} = {shown}")
    ax.set_yticks(range(len(order)), labels)
    ax.set_xlabel("Contribution to the log-odds of disease")
    _x_grid_only(ax)
    return fig
