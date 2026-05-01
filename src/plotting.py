# src/plotting.py

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


MODEL_STYLES = {
    "Naive": {"color": "#E53935", "ls": "--"},
    "SeasonalNaive": {"color": "#1E88E5", "ls": "--"},
    "AutoETS": {"color": "#43A047", "ls": "--"},
    "Chronos-t5-mini": {"color": "#FB8C00", "ls": "-."},
    "LightGBM": {"color": "#7B1FA2", "ls": "--"},
    "LightGBM_Base": {"color": "#7B1FA2", "ls": "--"},
    "LightGBM-Rich": {"color": "#512DA8", "ls": "--"},
    "LightGBM_Enhanced": {"color": "#512DA8", "ls": "--"},
    "LightGBM-Cat": {"color": "#311B92", "ls": "--"},
    "LightGBM_Enhanced_Static": {"color": "#311B92", "ls": "--"},
    "NHITS": {"color": "#F4511E", "ls": "--"},
}


def _style_for(model: str) -> dict:
    return MODEL_STYLES.get(model, {"color": "#555555", "ls": "--"})


def select_top_series(
    df: pd.DataFrame,
    n: int = 1,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> list[str]:
    return (
        df.groupby(id_col)[target_col]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index
        .tolist()
    )


def plot_sample_series_grid(
    df: pd.DataFrame,
    unique_ids: Iterable[str],
    title: str = "Sample Series — Daily Sales",
    ncols: int = 2,
    figsize: tuple[int, int] = (16, 6),
):
    unique_ids = list(unique_ids)
    nrows = int(np.ceil(len(unique_ids) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, uid in zip(axes, unique_ids):
        s = df[df["unique_id"] == uid].sort_values("ds")
        ax.plot(s["ds"], s["y"], linewidth=0.8, color="steelblue")
        ax.set_title(uid, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Units sold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    for ax in axes[len(unique_ids):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_dayofweek_profile(
    df: pd.DataFrame,
    title: str = "Average Daily Sales by Day of Week",
):
    dow_profile = df.assign(dow=df["ds"].dt.dayofweek).groupby("dow")["y"].mean()
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(day_labels, dow_profile.values, color="steelblue", alpha=0.85)
    ax.axhline(
        dow_profile.mean(),
        color="tomato",
        linestyle="--",
        linewidth=1,
        label="Panel mean",
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Mean units sold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    return fig, ax, dow_profile


def plot_histogram(
    values,
    title: str,
    xlabel: str,
    ylabel: str = "Number of series",
    bins: int = 40,
    threshold: float | None = None,
    threshold_label: str | None = None,
    figsize: tuple[int, int] = (10, 4),
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, color="steelblue", alpha=0.85, edgecolor="white")

    if threshold is not None:
        ax.axvline(
            threshold,
            color="tomato",
            linestyle="--",
            linewidth=1.5,
            label=threshold_label or "Threshold",
        )
        ax.legend(fontsize=9)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_cv_windows(
    series: pd.Series,
    cutoffs: Iterable[pd.Timestamp],
    horizon: int,
    title: str | None = None,
    lookback_days: int = 60,
):
    cutoffs = list(pd.to_datetime(cutoffs))
    plot_start = cutoffs[0] - pd.Timedelta(days=lookback_days)
    sample_plot = series[series.index >= plot_start]

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#009688"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sample_plot.index, sample_plot.values, color="#555", linewidth=0.9, zorder=2)

    for i, cutoff in enumerate(cutoffs):
        color = colors[i % len(colors)]
        horizon_end = cutoff + pd.Timedelta(days=horizon)
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
        ax.axvspan(
            cutoff,
            horizon_end,
            alpha=0.12,
            color=color,
            label=f"Window {i+1}: {cutoff.date()} + {horizon}d",
        )

    ax.set_title(title or "Cross-Validation Windows", fontsize=11)
    ax.set_ylabel("Units sold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_forecast_overlay(
    actuals_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    unique_id: str,
    cutoff=None,
    models: Iterable[str] | None = None,
    lookback_days: int = 42,
    title: str | None = None,
    show_intervals: bool = True,
    interval_alpha: float = 0.15,
    figsize: tuple[int, int] = (14, 4),
):
    if cutoff is None:
        cutoff = forecasts_df["cutoff"].max()
    cutoff = pd.to_datetime(cutoff)

    if models is None:
        models = forecasts_df["model"].dropna().unique().tolist()
    else:
        models = list(models)

    actuals = (
        actuals_df[actuals_df["unique_id"] == unique_id]
        .sort_values("ds")
        .set_index("ds")["y"]
    )

    plot_start = cutoff - pd.Timedelta(days=lookback_days)
    act_plot = actuals[actuals.index >= plot_start]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        act_plot.index,
        act_plot.values,
        color="#333333",
        linewidth=1.0,
        label="Actual",
        zorder=3,
    )

    for model in models:
        style = _style_for(model)
        fcast = forecasts_df[
            (forecasts_df["unique_id"] == unique_id)
            & (pd.to_datetime(forecasts_df["cutoff"]) == cutoff)
            & (forecasts_df["model"] == model)
        ].sort_values("ds")

        if fcast.empty:
            continue

        ax.plot(
            fcast["ds"],
            fcast["y_hat"],
            color=style["color"],
            linewidth=1.5,
            linestyle=style["ls"],
            label=model,
            zorder=4,
        )

        has_interval = (
            show_intervals
            and {"lo_80", "hi_80"}.issubset(fcast.columns)
            and fcast["lo_80"].notna().any()
            and fcast["hi_80"].notna().any()
        )

        if has_interval:
            ax.fill_between(
                fcast["ds"],
                fcast["lo_80"],
                fcast["hi_80"],
                alpha=interval_alpha,
                color=style["color"],
            )

    ax.axvline(cutoff, color="#999999", linestyle=":", linewidth=1)
    ax.set_title(title or f"Forecast Overlay — {unique_id}", fontsize=11)
    ax.set_ylabel("Units sold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_metric_leaderboard(
    df: pd.DataFrame,
    metric: str = "wMAPE",
    title: str | None = None,
    lower_is_better: bool = True,
    figsize: tuple[int, int] = (9, 4),
):
    if {"metric", "score"}.issubset(df.columns):
        plot_df = (
            df[df["metric"] == metric][["model", "score"]]
            .rename(columns={"score": metric})
            .dropna()
        )
    else:
        plot_df = df[["model", metric]].dropna()

    plot_df = plot_df.sort_values(metric, ascending=lower_is_better)
    colors = [_style_for(m)["color"] for m in plot_df["model"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(plot_df["model"], plot_df[metric], color=colors, edgecolor="white")
    ax.set_xlabel(f"{metric} ({'lower' if lower_is_better else 'higher'} = better)")
    ax.set_title(title or f"{metric} Leaderboard", fontsize=10)
    ax.invert_yaxis()

    offset = plot_df[metric].max() * 0.01 if plot_df[metric].max() else 0.001
    for bar, val in zip(bars, plot_df[metric]):
        label = f"{val:.4f}" if abs(val) < 10 else f"{val:,.1f}"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_interval_grid(
    actuals_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    unique_id: str,
    models: Iterable[str],
    cutoff=None,
    lookback_days: int = 42,
    title: str | None = None,
    ncols: int = 3,
    figsize: tuple[int, int] = (18, 8),
):
    models = list(models)
    if cutoff is None:
        cutoff = forecasts_df["cutoff"].max()
    cutoff = pd.to_datetime(cutoff)

    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    actuals = (
        actuals_df[actuals_df["unique_id"] == unique_id]
        .sort_values("ds")
        .set_index("ds")["y"]
    )

    plot_start = cutoff - pd.Timedelta(days=lookback_days)
    act_plot = actuals[actuals.index >= plot_start]

    for ax, model in zip(axes, models):
        style = _style_for(model)
        ax.plot(
            act_plot.index,
            act_plot.values,
            color="#333",
            linewidth=1.0,
            label="Actual",
            zorder=3,
        )

        fcast = forecasts_df[
            (forecasts_df["unique_id"] == unique_id)
            & (pd.to_datetime(forecasts_df["cutoff"]) == cutoff)
            & (forecasts_df["model"] == model)
        ].sort_values("ds")

        if not fcast.empty:
            ax.plot(
                fcast["ds"],
                fcast["y_hat"],
                color=style["color"],
                linewidth=1.5,
                linestyle=style["ls"],
                label=model,
                zorder=4,
            )

            if (
                {"lo_80", "hi_80"}.issubset(fcast.columns)
                and fcast["lo_80"].notna().any()
            ):
                ax.fill_between(
                    fcast["ds"],
                    fcast["lo_80"],
                    fcast["hi_80"],
                    alpha=0.18,
                    color=style["color"],
                )

        ax.axvline(cutoff, color="#aaaaaa", linestyle=":", linewidth=0.8)
        ax.set_title(model, fontsize=10)
        ax.set_ylabel("Units")
        ax.tick_params(axis="x", labelsize=7)

    for ax in axes[len(models):]:
        ax.axis("off")

    fig.suptitle(title or f"Prediction Intervals — {unique_id}", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_interval_width_distribution(
    forecasts_df: pd.DataFrame,
    models: Iterable[str],
    title: str = "Interval Width Distribution by Model",
    figsize: tuple[int, int] = (12, 4),
):
    interval_df = forecasts_df[
        forecasts_df["model"].isin(models)
        & forecasts_df["lo_80"].notna()
        & forecasts_df["hi_80"].notna()
    ].copy()

    interval_df["width"] = interval_df["hi_80"] - interval_df["lo_80"]

    fig, ax = plt.subplots(figsize=figsize)

    for model in sorted(models):
        style = _style_for(model)
        widths = interval_df[interval_df["model"] == model]["width"]

        if widths.empty:
            continue

        ax.hist(
            widths.clip(upper=widths.quantile(0.99)),
            bins=60,
            alpha=0.4,
            color=style["color"],
            label=model,
            density=True,
        )

    ax.set_xlabel("Interval width (hi_80 − lo_80)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

    medians = interval_df.groupby("model")["width"].median().sort_values()
    return fig, ax, medians


def plot_metric_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annotate: bool = True,
    figsize: tuple[int, int] = (9, 7),
):
    plot_df = df.dropna(subset=[x, y]).copy()

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in plot_df.iterrows():
        style = _style_for(row["model"])
        ax.scatter(
            row[x],
            row[y],
            s=160,
            color=style["color"],
            zorder=4,
            edgecolors="white",
            linewidth=1.2,
        )

        if annotate:
            ax.annotate(
                row["model"],
                xy=(row[x], row[y]),
                xytext=(8, 3),
                textcoords="offset points",
                fontsize=9,
                color=style["color"],
                fontweight="bold",
            )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig, ax
