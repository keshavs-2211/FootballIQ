
"""FootballIQ match overview: measured, interpretable analytics."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytics.common import PALETTE, savefig

BG = "#101821"
PANEL = "#182430"
TEXT = "#eef3f7"
MUTED = "#9db0be"
GRID = "#314251"
TEAMS = ("Team A", "Team B")


def style_axis(ax, title, xlabel=None):
    ax.set_facecolor(PANEL)
    ax.set_title(title, loc="left", color=TEXT,
                 fontsize=12, weight="bold", pad=14)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, alpha=0.45)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)


def empty(ax, message):
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha="center", va="center",
        color=MUTED, fontsize=11,
    )


def generate(tracks, team_stats, shape_summary,
             clip, fps, data, root):

    root = Path(root)
    observed = data[data["is_observed"]].copy()
    observed = observed[observed["team"].isin(TEAMS)]
    observed = observed[
        observed["pitch_x_m"].between(0, 105)
        & observed["pitch_y_m"].between(0, 68)
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 10),
        facecolor=BG
    )
    fig.subplots_adjust(
        left=0.09, right=0.96,
        top=0.81, bottom=0.13,
        wspace=0.28, hspace=0.45
    )

    fig.text(
        0.09, 0.95, "FOOTBALLIQ",
        color="#61ceb4", fontsize=13,
        weight="bold"
    )
    fig.text(
        0.09, 0.906,
        f"{clip.replace('_', ' ').title()} — Match analysis",
        color=TEXT, fontsize=22, weight="bold"
    )
    fig.text(
        0.09, 0.862,
        "Observed tracking data  |  Clip-level estimates  |  "
        "Not full-match physical statistics",
        color=MUTED, fontsize=10
    )

    # --------------------------------------------------
    # A. Movement distance normalized by measured time
    # --------------------------------------------------
    ax = axes[0, 0]
    style_axis(
        ax, "Movement per measured minute",
        "Metres per minute of valid movement"
    )

    stats = tracks.copy()
    time = pd.to_numeric(
        stats["measured_movement_time_s"], errors="coerce"
    )
    distance = pd.to_numeric(
        stats["distance_m_est"], errors="coerce"
    )

    stats["metres_per_min"] = np.where(
        time >= 5,
        distance / time.clip(lower=1) * 60,
        np.nan
    )

    stats = stats[
        stats["team"].isin(TEAMS)
        & stats["metres_per_min"].notna()
        & np.isfinite(stats["metres_per_min"])
    ]

    stats = stats.nlargest(
        8, "metres_per_min"
    ).sort_values("metres_per_min")

    if not stats.empty:
        labels = [
            f"Track {int(row.track_id)}"
            for row in stats.itertuples()
        ]
        values = stats["metres_per_min"].to_numpy()

        ax.barh(
            labels, values,
            color=[PALETTE[t] for t in stats["team"]],
            height=0.66
        )
        ax.grid(axis="x", color=GRID, alpha=0.45)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(values) * 1.22)

        for i, value in enumerate(values):
            ax.text(
                value + max(values) * 0.02, i,
                f"{value:.1f}",
                color=TEXT, va="center", fontsize=9
            )
    else:
        empty(ax, "Insufficient movement observations")

    # --------------------------------------------------
    # B. Pitch thirds (not attacking/defending thirds)
    # --------------------------------------------------
    ax = axes[0, 1]
    style_axis(
        ax, "Observed activity by pitch zone",
        "Share of each team's observed positions (%)"
    )

    zone_names = [
        "Left third", "Middle third", "Right third"
    ]

    x = np.arange(3)
    width = 0.34
    has_data = False

    for index, team in enumerate(TEAMS):
        group = observed[observed["team"] == team]

        if group.empty:
            values = np.zeros(3)
        else:
            zone = pd.cut(
                group["pitch_x_m"],
                bins=[-0.001, 35, 70, 105.001],
                labels=False,
                include_lowest=True
            )
            counts = zone.value_counts()
            values = np.array([
                counts.get(i, 0) for i in range(3)
            ], dtype=float)
            values = values / max(values.sum(), 1) * 100
            has_data = True

        offset = (-0.5 if index == 0 else 0.5) * width

        ax.bar(
            x + offset, values,
            width=width,
            color=PALETTE[team],
            label=team
        )

    ax.set_xticks(x, zone_names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of observed positions", color=MUTED)

    if has_data:
        ax.legend(
            frameon=False, labelcolor=TEXT,
            fontsize=9
        )
    else:
        empty(ax, "No observed pitch positions")

    # --------------------------------------------------
    # C. Visible tracked players over video time
    # --------------------------------------------------
    ax = axes[1, 0]
    style_axis(
        ax, "Tracking coverage over time",
        "Video time (seconds)"
    )
    ax.set_ylabel("Visible tracked players", color=MUTED)

    if not observed.empty and fps > 0:
        bin_frames = max(1, int(round(fps * 2)))
        coverage = observed.copy()
        coverage["time_bin"] = (
            coverage["frame"] // bin_frames
        )

        # Count unique tracks per frame, then average
        # within each two-second interval.
        counts = (
            coverage.groupby(["time_bin", "frame", "team"])
            ["track_id"]
            .nunique()
            .reset_index(name="visible")
        )

        for team in TEAMS:
            g = counts[counts["team"] == team]
            if g.empty:
                continue

            # Reindex to include intervals with no observations.
            max_bin = int(
                observed["frame"].max() // bin_frames
            )
            trend = (
                g.groupby("time_bin")["visible"]
                .mean()
                .reindex(range(max_bin + 1), fill_value=0)
            )
            times = (
                trend.index.to_numpy()
                * bin_frames / fps
            )
            ax.plot(
                times, trend.to_numpy(),
                color=PALETTE[team],
                linewidth=2,
                label=team
            )

        ax.legend(
            frameon=False, labelcolor=TEXT,
            fontsize=9
        )
        ax.set_ylim(bottom=0)
    else:
        empty(ax, "No observed tracking data")

    # --------------------------------------------------
    # D. Average width and depth of visible team shape
    # --------------------------------------------------
    ax = axes[1, 1]
    style_axis(
        ax, "Team spread when sufficiently visible",
        "Average spread (metres)"
    )

    shape = shape_summary.copy()
    shape = shape.set_index("team")

    labels = ["Width", "Depth"]
    x = np.arange(2)
    width = 0.34
    has_shape = False

    for i, team in enumerate(TEAMS):
        values = []

        for field in (
            "average_width_m",
            "average_depth_m"
        ):
            value = (
                shape.loc[team, field]
                if team in shape.index
                else np.nan
            )
            value = pd.to_numeric(value, errors="coerce")
            values.append(
                float(value) if pd.notna(value) else 0
            )

        if any(values):
            has_shape = True

        offset = (-0.5 if i == 0 else 0.5) * width

        ax.bar(
            x + offset, values,
            width=width,
            color=PALETTE[team],
            label=team
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Metres", color=MUTED)

    if has_shape:
        ax.legend(
            frameon=False, labelcolor=TEXT,
            fontsize=9
        )
    else:
        empty(ax, "Not enough simultaneous players")

    fig.text(
        0.09, 0.045,
        "Movement rates use measured track segments. "
        "Pitch zones describe camera-visible locations, "
        "not attacking direction. Tracking gaps and "
        "calibration quality affect every estimate.",
        color=MUTED, fontsize=9
    )

    savefig(
        fig,
        root / "dashboard" / "analytics_overview.png"
    )
