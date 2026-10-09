#!/usr/bin/env python3
"""
compile_figures.py - Compiles publication-ready figures for the HotMobile 2027 paper.
Outputs:
  - figures/heartbeat_stats.png (Figure 2 in paper)
  - figures/fig_throughput_minmax_accessible.png (Figure 3 in paper)
  - figures/bidirectional_3panel.png (Figure 4 in paper)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from helpers import (
    DATA_DIR,
    FIGURES_DIR,
    HEADSET_IPS,
    HEADSET_LABELS,
    SESSION_PHASES,
    BIN_SEC,
    COLOR_UP,
    COLOR_DOWN,
    COLOR_FIT,
    COLOR_REF,
    COLOR_SERVER_PING,
    COLOR_CLIENT_PING,
    set_publication_style,
    extract_game_traffic,
    extract_heartbeats_and_rtt,
    compute_throughput_timeseries,
    fit_downstream_gmm,
    fit_payload_distributions,
)


def plot_figure_2_heartbeat_stats(df_rtt: pd.DataFrame, df_loss_or_all: pd.DataFrame, out_dir: Path):
    """
    Figure 2: Passive heartbeat probe extraction and directional latency characterization:
    (a) Directional RTT & Processing Asymmetry Boxplots (isolating 3.8 ms floor and 7/12 ms cohorts)
    (b) Directional Packet Loss Asymmetry (highlighting downlink Wi-Fi AP queue contention)
    """
    print("Generating Figure 2: figures/heartbeat_stats.png ...")
    set_publication_style()

    if df_loss_or_all is not None and "loss_pct" in df_loss_or_all.columns:
        df_loss = df_loss_or_all
    else:
        _, df_loss = extract_heartbeats_and_rtt()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 4.0), dpi=300)

    # Sequential Headset IDs 1 to 9
    x_labels = [f"ID {i+1}" for i in range(len(HEADSET_IPS))]
    x_indices = np.arange(len(HEADSET_IPS))
    width = 0.35

    # -----------------------------------------------------------------------
    # Subplot (a): Directional RTT Distribution & Server Cohorts (Boxplots)
    # -----------------------------------------------------------------------
    s2c_data = [df_rtt[(df_rtt["headset_ip"] == ip) & (df_rtt["direction"] == "s2c")]["rtt_ms"].values for ip in HEADSET_IPS]
    c2s_data = [df_rtt[(df_rtt["headset_ip"] == ip) & (df_rtt["direction"] == "c2s")]["rtt_ms"].values for ip in HEADSET_IPS]

    # Server Probe (s2c) boxplots in Vermilion
    ax1.boxplot(
        s2c_data,
        positions=x_indices - width / 2,
        widths=width * 0.78,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=1.5),
        boxprops=dict(facecolor=COLOR_DOWN, alpha=0.85, edgecolor="black", lw=0.9),
        whiskerprops=dict(color="black", lw=1.0),
        capprops=dict(color="black", lw=1.0),
        zorder=3
    )

    # Client Probe (c2s) boxplots in Blue
    ax1.boxplot(
        c2s_data,
        positions=x_indices + width / 2,
        widths=width * 0.78,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=1.5),
        boxprops=dict(facecolor=COLOR_UP, alpha=0.85, edgecolor="black", lw=0.9),
        whiskerprops=dict(color="black", lw=1.0),
        capprops=dict(color="black", lw=1.0),
        zorder=3
    )

    # Wi-Fi hop median (s2c) and WAN+server cohort reference lines (c2s)
    wifi_med = df_rtt[df_rtt["direction"] == "s2c"]["rtt_ms"].median()
    ax1.axhline(wifi_med, color=COLOR_DOWN, ls=":", lw=1.5, alpha=0.95, zorder=2)
    ax1.axhline(7.06, color=COLOR_UP, ls="--", lw=1.2, alpha=0.9, zorder=2)
    ax1.axhline(12.06, color=COLOR_UP, ls="-.", lw=1.2, alpha=0.9, zorder=2)

    bbox_props = dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85)
    ax1.text(0.15, wifi_med + 0.35, f"Wi-Fi hop median (~{wifi_med:.1f} ms)",
             color=COLOR_DOWN, fontsize=8, ha="left", va="bottom", fontweight="bold", bbox=bbox_props, zorder=5)
    ax1.text(0.15, 7.06 + 0.35, "WAN+server cohort 1 (~7 ms)",
             color=COLOR_UP, fontsize=8, ha="left", va="bottom", fontweight="bold", bbox=bbox_props, zorder=5)
    ax1.text(0.15, 12.06 + 0.35, "WAN+server cohort 2 (~12 ms)",
             color=COLOR_UP, fontsize=8, ha="left", va="bottom", fontweight="bold", bbox=bbox_props, zorder=5)

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Round-Trip Time (ms)", fontweight="bold")
    ax1.set_title("(a) Directional RTT Decomposition", pad=6, fontweight="bold")
    ax1.set_ylim(0, 24.0)
    ax1.grid(True, ls=":", alpha=0.6, axis="y", zorder=0)

    legend_elements_a = [
        Patch(facecolor=COLOR_DOWN, alpha=0.85, edgecolor="black", label="Server probe (s2c): Wi-Fi hop + HMD stack"),
        Patch(facecolor=COLOR_UP, alpha=0.85, edgecolor="black", label="Client probe (c2s): WAN + server"),
    ]
    ax1.legend(handles=legend_elements_a, loc="upper right", fontsize=9, framealpha=0.92)

    # -----------------------------------------------------------------------
    # Subplot (b): Directional Loss Asymmetry (Wireless AP Contention)
    # -----------------------------------------------------------------------
    loss_s2c = []
    loss_c2s = []
    for ip in HEADSET_IPS:
        s_row = df_loss[(df_loss["headset_ip"] == ip) & (df_loss["direction"] == "s2c")]
        c_row = df_loss[(df_loss["headset_ip"] == ip) & (df_loss["direction"] == "c2s")]
        loss_s2c.append(s_row["loss_pct"].values[0] if len(s_row) > 0 else 0)
        loss_c2s.append(c_row["loss_pct"].values[0] if len(c_row) > 0 else 0)

    # Identical colors: Server probe in Vermilion, Client probe in Blue
    b_s = ax2.bar(x_indices - width / 2, loss_s2c, width * 0.78, color=COLOR_DOWN, alpha=0.85, edgecolor="black", lw=0.9,
                  label="Server probe (s2c) loss", zorder=3)
    b_c = ax2.bar(x_indices + width / 2, loss_c2s, width * 0.78, color=COLOR_UP, alpha=0.85, edgecolor="black", lw=0.9,
                  label="Client probe (c2s) loss", zorder=3)

    # Label prominent loss percentages
    for rect in b_s:
        h = rect.get_height()
        if h > 1.0:
            ax2.annotate(f"{h:.1f}%",
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, fontweight="bold", color=COLOR_DOWN)

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("Headset ID", fontweight="bold")
    ax2.set_ylabel("Probe Loss Rate (%)", fontweight="bold")
    ax2.set_title("(b) Directional Heartbeat Loss (Unanswered Probes)", pad=6, fontweight="bold")
    ax2.set_ylim(0, 20.0)
    ax2.grid(True, ls=":", alpha=0.6, axis="y", zorder=0)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    out_png = out_dir / "heartbeat_stats.png"
    out_pdf = out_dir / "heartbeat_stats.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"  -> Saved \"heartbeat_stats.png\" and \"heartbeat_stats.pdf\"")


def plot_figure_3_throughput(df_all: pd.DataFrame, out_dir: Path):
    """
    Figure 3: Accessible 3-Panel Throughput Plot with Min/Max Envelope & Ground-Truth Phases.
    (a) Total Aggregate Throughput
    (b) Upstream Per-Headset Throughput
    (c) Downstream Per-Headset Throughput (Stabilization during Lecture)
    """
    print("Generating Figure 3: figures/fig_throughput_minmax_accessible.png ...")
    set_publication_style()

    per_hs, stats_hs, total_agg, _ = compute_throughput_timeseries(df_all)

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 6.0), sharex=True, dpi=300)

    def add_scenario_shading(ax):
        for label, start, end, col in SESSION_PHASES:
            ax.axvspan(start, end, facecolor=col, alpha=0.7, edgecolor="none", zorder=0)
            ax.axvline(start, color="#475569", linestyle="--", linewidth=0.9, zorder=1, alpha=0.7)
        ax.axvline(SESSION_PHASES[-1][2], color="#475569", linestyle="--", linewidth=0.9, zorder=1, alpha=0.7)

    # (a) Total Aggregate Throughput
    ax0 = axes[0]
    add_scenario_shading(ax0)
    agg_down = total_agg[total_agg["direction"] == "downstream"]
    agg_up = total_agg[total_agg["direction"] == "upstream"]

    ax0.plot(agg_down["t_min"], agg_down["kbps"], color=COLOR_DOWN, lw=1.8, label="Downstream (Server → HMDs)")
    ax0.plot(agg_up["t_min"], agg_up["kbps"], color=COLOR_UP, lw=1.8, label="Upstream (HMDs → Server)")
    ax0.set_ylabel("Aggregate (kbps)")
    ax0.set_title("(a) Total Aggregate Throughput", pad=4)
    ax0.grid(True, linestyle=":", alpha=0.6, zorder=1)
    ax0.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # (b) Upstream Per-Headset (Min, Max, Mean, ±1 Std)
    ax1 = axes[1]
    add_scenario_shading(ax1)
    up_s = stats_hs[stats_hs["direction"] == "upstream"]

    ax1.fill_between(up_s["t_min"], up_s["min"], up_s["max"], color=COLOR_UP, alpha=0.15, label="[Min, Max] Range")
    ax1.fill_between(up_s["t_min"], (up_s["mean"] - up_s["std"]).clip(lower=0), up_s["mean"] + up_s["std"],
                     color=COLOR_UP, alpha=0.30, label="Mean ± 1σ")
    ax1.plot(up_s["t_min"], up_s["mean"], color=COLOR_UP, lw=1.6, label="Mean (N = 9)")
    ax1.set_ylabel("Per-HMD kbps")
    ax1.set_title("(b) Upstream Per-Headset Throughput", pad=4)
    ax1.grid(True, linestyle=":", alpha=0.6, zorder=1)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # (c) Downstream Per-Headset Throughput
    ax2 = axes[2]
    add_scenario_shading(ax2)
    down_s = stats_hs[stats_hs["direction"] == "downstream"]

    ax2.fill_between(down_s["t_min"], down_s["min"], down_s["max"], color=COLOR_DOWN, alpha=0.15, label="[Min, Max] Range")
    ax2.fill_between(down_s["t_min"], (down_s["mean"] - down_s["std"]).clip(lower=0), down_s["mean"] + down_s["std"],
                     color=COLOR_DOWN, alpha=0.30, label="Mean ± 1σ")
    ax2.plot(down_s["t_min"], down_s["mean"], color=COLOR_DOWN, lw=1.6, label="Mean (N = 9)")
    ax2.set_ylabel("Per-HMD kbps")
    ax2.set_xlabel("Minutes from start of measurement")
    ax2.set_title("(c) Downstream Per-Headset Throughput (Stabilization during Lecture)", pad=4)
    ax2.grid(True, linestyle=":", alpha=0.6, zorder=1)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Top Scenario Legend
    phase_patches = [Patch(facecolor=col, edgecolor="gray", alpha=0.8, label=label) for label, _, _, col in SESSION_PHASES]
    fig.legend(handles=phase_patches, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05),
               frameon=False, fontsize=9)

    plt.xlim(0, 43.4)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    out_png = out_dir / "fig_throughput_minmax_accessible.png"
    out_pdf = out_dir / "fig_throughput_minmax_accessible.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"  -> Saved \"fig_throughput_minmax_accessible.png\" and \"fig_throughput_minmax_accessible.pdf\"")


def plot_figure_4_bidirectional_models(df_all: pd.DataFrame, out_dir: Path):
    """
    Figure 4: 3-Panel Bidirectional Packet Timing and Payload Characterization:
    (a) Inter-Arrival PDF (Overlaid with 4-Component GMM)
    (b) Inter-Arrival CDF (Overlaid with GMM CDF)
    (c) Payload Size Distribution (Overlaid with Log-Normal Fit)
    """
    print("Generating Figure 4: figures/bidirectional_3panel.png ...")
    set_publication_style()

    # 1. Fit GMM on Downstream IPT
    _, _, delta_down_clean, gmm_params = fit_downstream_gmm(df_all)
    means, stds, weights = gmm_params

    # 2. Extract Upstream IPT
    df_state = df_all[~df_all["is_heartbeat"]].copy()
    df_state["delta_t_ms"] = df_state.groupby(["headset_ip", "direction"])["timestamp"].diff() * 1000.0
    dt_up_clean = df_state[df_state["direction"] == "upstream"]["delta_t_ms"].dropna()
    dt_up_clean = dt_up_clean[(dt_up_clean >= 5.0) & (dt_up_clean <= 75.0)].values
    dt_down_clean = delta_down_clean[(delta_down_clean >= 5.0) & (delta_down_clean <= 75.0)]

    # 3. Fit Payload Distributions
    lognorm_params, s_down, s_up = fit_payload_distributions(df_all)
    shape_d, loc_d, scale_d = lognorm_params

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.5, 3.5), dpi=300)

    # =========================================================================
    # (a) Inter-Arrival PDF
    # =========================================================================
    x_arr = np.linspace(5.0, 75.0, 2500)
    kde_down = gaussian_kde(dt_down_clean, bw_method=0.009)
    kde_up = gaussian_kde(dt_up_clean, bw_method=0.035)

    y_down_pdf = kde_down(x_arr)
    y_up_pdf = kde_up(x_arr)

    y_gmm_pdf = np.zeros_like(x_arr)
    for m, s, w in zip(means, stds, weights):
        y_gmm_pdf += w * stats.norm.pdf(x_arr, loc=m, scale=s)

    line_down, = ax1.plot(x_arr, y_down_pdf, color=COLOR_DOWN, lw=1.8, label="Downstream (Server→HMD)")
    line_up, = ax1.plot(x_arr, y_up_pdf, color=COLOR_UP, lw=1.5, ls="--", label="Upstream (HMD→Server)")
    line_fit, = ax1.plot(x_arr, y_gmm_pdf, color=COLOR_FIT, lw=1.2, ls=":", label="Downstream 4-GMM Fit")

    ax1.set_ylim(0.0, 1.05)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    for idx in [0, 1, 3]:
        m = means[idx]
        freq = 1000.0 / m
        ax1.axvline(m, color=COLOR_REF, ls=":", lw=0.9, alpha=0.7)
        ax1.text(m + 1.2, 0.98, f"Mode {idx+1}\n~{freq:.0f} Hz\n({m:.1f} ms)", fontsize=7, color="#1E293B", va="top")

    ax1.set_xlabel(r"Inter-Arrival Time $\tau$ [ms]", fontsize=9)
    ax1.set_ylabel("Probability Density", fontsize=9)
    ax1.set_title("(a) Inter-Arrival PDF", fontsize=10, pad=4)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_xlim(5, 75)

    # =========================================================================
    # (b) Inter-Arrival CDF
    # =========================================================================
    sorted_down = np.sort(dt_down_clean)
    cdf_down = np.arange(1, len(sorted_down) + 1) / len(sorted_down)

    sorted_up = np.sort(dt_up_clean)
    cdf_up = np.arange(1, len(sorted_up) + 1) / len(sorted_up)

    ax2.plot(sorted_down, cdf_down, color=COLOR_DOWN, lw=1.8)
    ax2.plot(sorted_up, cdf_up, color=COLOR_UP, lw=1.6, ls="--")

    y_gmm_cdf = np.zeros_like(x_arr)
    for m, s, w in zip(means, stds, weights):
        y_gmm_cdf += w * stats.norm.cdf(x_arr, loc=m, scale=s)
    ax2.plot(x_arr, y_gmm_cdf, color=COLOR_FIT, lw=1.2, ls=":")

    for idx in [0, 1, 3]:
        m = means[idx]
        ax2.axvline(m, color=COLOR_REF, ls=":", lw=0.9, alpha=0.7)

    ax2.axhline(0.5, color="gray", linestyle="-.", lw=0.8, alpha=0.5)
    ax2.set_xlabel(r"Inter-Arrival Threshold $x$ [ms]", fontsize=9)
    ax2.set_ylabel(r"Cumulative Prob. $P(\tau \leq x)$", fontsize=9)
    ax2.set_title("(b) Inter-Arrival CDF", fontsize=10, pad=4)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_xlim(5, 75)
    ax2.set_ylim(0, 1.02)

    # =========================================================================
    # (c) Payload Size Distribution
    # =========================================================================
    s_down_clean = s_down[(s_down > 20) & (s_down <= 600)]
    s_up_clean = s_up[(s_up > 20) & (s_up <= 600)]
    x_pay = np.linspace(20, 600, 1000)

    kde_pay_down = gaussian_kde(s_down_clean, bw_method=0.08)
    kde_pay_up = gaussian_kde(s_up_clean, bw_method=0.08)

    y_pay_down = kde_pay_down(x_pay)
    y_pay_up = kde_pay_up(x_pay)
    y_pay_lognorm = stats.lognorm.pdf(x_pay, s=shape_d, loc=loc_d, scale=scale_d)

    ax3.plot(x_pay, y_pay_down, color=COLOR_DOWN, lw=1.8)
    ax3.plot(x_pay, y_pay_up, color=COLOR_UP, lw=1.6, ls="--")
    ax3.plot(x_pay, y_pay_lognorm, color=COLOR_FIT, lw=1.2, ls=":")

    # Medians over all non-heartbeat payloads (matches values reported in the paper text)
    med_down = np.median(s_down)
    med_up = np.median(s_up)

    max_y3 = max(np.max(y_pay_down), np.max(y_pay_up))
    ax3.set_ylim(0, max_y3 * 1.25)

    ax3.axvline(med_down, color=COLOR_DOWN, ls=":", lw=0.9, alpha=0.7)
    ax3.text(med_down + 8, max_y3 * 0.95, f"Down Med:\n{med_down:.0f} B", fontsize=7, color=COLOR_DOWN, va="top")

    ax3.axvline(med_up, color=COLOR_UP, ls=":", lw=0.9, alpha=0.7)
    ax3.text(med_up + 6, max_y3 * 0.95, f"Up Med:\n{med_up:.0f} B", fontsize=7, color=COLOR_UP, va="top")

    ax3.set_xlabel(r"Payload Size $S$ [Bytes]", fontsize=9)
    ax3.set_ylabel("Probability Density", fontsize=9)
    ax3.set_title("(c) Payload Size Distribution", fontsize=10, pad=4)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.set_xlim(20, 600)

    # Shared Top Legend
    fig.legend(
        handles=[line_down, line_up, line_fit],
        labels=[
            "Downstream Empirical Trace",
            "Upstream Empirical Trace",
            "Downstream 4-GMM / Log-Normal Fit",
        ],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.24)

    out_png = out_dir / "bidirectional_3panel.png"
    out_pdf = out_dir / "bidirectional_3panel.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"  -> Saved \"bidirectional_3panel.png\" and \"bidirectional_3panel.pdf\"")


def main():
    parser = argparse.ArgumentParser(description="Compile publication figures for HotMobile 2027 paper")
    parser.add_argument("--out", default=str(FIGURES_DIR), help="Output directory for figures (default: figures/)")
    parser.add_argument("--cache", default=str(DATA_DIR / "classified_traffic.csv"), help="Path to cached classified traffic CSV")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading traffic dataset...")
    df_all = extract_game_traffic(cache_csv=Path(args.cache))

    print(f"Extracting heartbeats & RTTs...")
    df_rtt, df_loss = extract_heartbeats_and_rtt()

    # Generate all three paper figures
    plot_figure_2_heartbeat_stats(df_rtt, df_loss, out_dir)
    plot_figure_3_throughput(df_all, out_dir)
    plot_figure_4_bidirectional_models(df_all, out_dir)

    print("\nAll paper figures compiled successfully!")


if __name__ == "__main__":
    main()
