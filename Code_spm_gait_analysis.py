# -*- coding: utf-8 -*-
"""
1D Statistical Parametric Mapping (SPM) for Gait Kinematics Analysis
====================================================================
This script performs 1D cluster permutation tests (two-sample and paired)
on lower-extremity gait cycle kinematics (Hip, Knee, Ankle) and generates
publication-ready 3-panel plots overlaid with Perry's 8 gait subphases.
"""

import os
import re
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

def _norm_path(p: str) -> str:
    """Resolve relative paths relative to this script's directory."""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    return os.path.join(base, p)


# ---------------------------------------------------------------------------
# Perry's 8 Gait Subphases Boundaries (%)
# ---------------------------------------------------------------------------
PHASES_8 = {
    "IC":  (0, 2),
    "LR":  (2, 10),
    "MS":  (10, 30),
    "TS":  (30, 50),
    "PSw": (50, 60),
    "ISw": (60, 73),
    "MSw": (73, 87),
    "TSw": (87, 100),
}

STANCE_END = 60.0


# ---------------------------------------------------------------------------
# Plotting Style Configurations
# ---------------------------------------------------------------------------
def set_plot_style(font="Times New Roman", base_size=12):
    """Set global matplotlib parameters for scientific publication style."""
    mpl.rcParams["font.family"] = font
    mpl.rcParams["font.size"] = base_size
    mpl.rcParams["axes.titlesize"] = base_size + 2
    mpl.rcParams["axes.labelsize"] = base_size + 1
    mpl.rcParams["legend.fontsize"] = base_size - 1
    mpl.rcParams["xtick.labelsize"] = base_size - 1
    mpl.rcParams["ytick.labelsize"] = base_size - 1
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42


def save_figure(fig, outdir, stem, formats=("png",), dpi=600):
    """Save the figure in multiple specified formats and high DPI."""
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(outdir, f"{stem}.{fmt}")
        if fmt.lower() in ("png", "jpg", "jpeg", "tif", "tiff"):
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Data Input/Output and Resampling Helpers
# ---------------------------------------------------------------------------
def find_wave_cols(df: pd.DataFrame):
    """Find and sort gait cycle percentage columns (e.g., pct_GC_1 to pct_GC_101)."""
    cols = [c for c in df.columns if isinstance(c, str) and ("pct_GC" in c)]
    if not cols:
        raise ValueError("Gait cycle columns (pct_GC) not found in the Excel file.")
    
    def extract_number(c):
        match = re.search(r"(\d+)\s*$", c)
        return int(match.group(1)) if match else 10**9
    return sorted(cols, key=extract_number)


def row_to_wave(row, wcols, x_new):
    """Interpolate a single row's gait data to the target percentage points."""
    y = row[wcols].to_numpy(dtype=float)
    x_old = np.linspace(0, 100, len(wcols))
    if len(x_new) != len(x_old):
        y = np.interp(x_new, x_old, y)
    return y


def waves_trial_level(df, wcols, x_new):
    """Extract waves treating every single trial as an independent sample."""
    return np.stack([row_to_wave(r, wcols, x_new) for _, r in df.iterrows()], axis=0)


def waves_subject_level_TD(df, wcols, x_new):
    """Average trials within each subject first for Typically Developed (TD) group."""
    if "Subject" not in df.columns:
        raise ValueError("The 'Subject' column is required for subject-level TD analysis.")
    waves = []
    for _, group in df.groupby("Subject"):
        ys = np.stack([row_to_wave(r, wcols, x_new) for _, r in group.iterrows()], axis=0)
        waves.append(np.nanmean(ys, axis=0))
    return np.stack(waves, axis=0)


def waves_subject_level_CP(df, wcols, x_new):
    """Average trials within each unique patient session for Cerebral Palsy (CP) group."""
    key_cols = ["Patient ID", "No. of GA", "cycle"]
    for c in key_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' is required for subject-level CP analysis.")
    waves = []
    for _, group in df.groupby(key_cols):
        ys = np.stack([row_to_wave(r, wcols, x_new) for _, r in group.iterrows()], axis=0)
        waves.append(np.nanmean(ys, axis=0))
    return np.stack(waves, axis=0)


# ---------------------------------------------------------------------------
# 1D Cluster Permutation Statistical Inference Engine
# ---------------------------------------------------------------------------
def _find_clusters(supra: np.ndarray):
    """Find contiguous suprathreshold clusters from a boolean mask."""
    idx = np.where(supra)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    return [(int(r[0]), int(r[-1])) for r in runs]


def _cluster_mass(t_curve, start, end):
    """Calculate the mass (integral of absolute t-values) of a cluster."""
    return float(np.sum(np.abs(t_curve[start:end+1])))


def twosample_cluster_perm(YA, YB, alpha=0.01, n_perm=5000, seed=123, equal_var=False):
    """Perform a 1D independent two-sample t-test with cluster permutation."""
    rng = np.random.default_rng(seed)
    YA = np.asarray(YA, float)
    YB = np.asarray(YB, float)
    nA, q = YA.shape
    nB, _ = YB.shape

    # Compute observed t-curve
    if equal_var:
        vA = YA.var(axis=0, ddof=1)
        vB = YB.var(axis=0, ddof=1)
        sp2 = ((nA-1)*vA + (nB-1)*vB) / (nA+nB-2)
        denom = np.sqrt(sp2*(1/nA + 1/nB))
        t_obs = (YA.mean(axis=0) - YB.mean(axis=0)) / denom
        df = nA + nB - 2
    else:
        vA = YA.var(axis=0, ddof=1)
        vB = YB.var(axis=0, ddof=1)
        denom = np.sqrt(vA/nA + vB/nB)
        t_obs = (YA.mean(axis=0) - YB.mean(axis=0)) / denom
        df_point = (vA/nA + vB/nB)**2 / ((vA**2)/((nA**2)*(nA-1)) + (vB**2)/((nB**2)*(nB-1)))
        df = float(np.nanmin(df_point[np.isfinite(df_point)]))
        df = max(df, 1.0)

    # Pointwise critical threshold
    t_thr = float(stats.t.ppf(1 - alpha/2, df=df))
    supra = np.abs(t_obs) > t_thr
    clusters = [{"start": s, "end": e, "mass": _cluster_mass(t_obs, s, e)} for s, e in _find_clusters(supra)]

    # Permutation testing (shuffling group labels)
    Y = np.vstack([YA, YB])
    labels = np.array([0]*nA + [1]*nB)
    max_masses = np.zeros(n_perm, float)

    for p in range(n_perm):
        rng.shuffle(labels)
        gA = Y[labels==0]
        gB = Y[labels==1]
        if equal_var:
            vA = gA.var(axis=0, ddof=1); vB = gB.var(axis=0, ddof=1)
            sp2 = ((gA.shape[0]-1)*vA + (gB.shape[0]-1)*vB) / (gA.shape[0]+gB.shape[0]-2)
            den = np.sqrt(sp2*(1/gA.shape[0] + 1/gB.shape[0]))
            tp = (gA.mean(axis=0) - gB.mean(axis=0)) / den
        else:
            vA = gA.var(axis=0, ddof=1); vB = gB.var(axis=0, ddof=1)
            den = np.sqrt(vA/gA.shape[0] + vB/gB.shape[0])
            tp = (gA.mean(axis=0) - gB.mean(axis=0)) / den

        suprap = np.abs(tp) > t_thr
        mm = 0.0
        for s, e in _find_clusters(suprap):
            mm = max(mm, _cluster_mass(tp, s, e))
        max_masses[p] = mm

    for c in clusters:
        c["p_value"] = float((np.sum(max_masses >= c["mass"]) + 1) / (n_perm + 1))

    return {"t_obs": t_obs, "t_thr": t_thr, "clusters": clusters, "df": df}


def paired_cluster_perm(YR, YL, alpha=0.01, n_perm=5000, seed=123):
    """Perform a 1D paired samples t-test with sign-flip cluster permutation."""
    rng = np.random.default_rng(seed)
    YR = np.asarray(YR, float)
    YL = np.asarray(YL, float)
    D = YR - YL
    n, q = D.shape

    denom = D.std(axis=0, ddof=1) / np.sqrt(n)
    t_obs = D.mean(axis=0) / denom
    t_thr = float(stats.t.ppf(1 - alpha/2, df=n-1))

    supra = np.abs(t_obs) > t_thr
    clusters = [{"start": s, "end": e, "mass": _cluster_mass(t_obs, s, e)} for s, e in _find_clusters(supra)]

    # Permutation testing (random sign flipping)
    max_masses = np.zeros(n_perm, float)
    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=(n, 1))
        Dp = D * signs
        den = Dp.std(axis=0, ddof=1) / np.sqrt(n)
        tp = Dp.mean(axis=0) / den
        suprap = np.abs(tp) > t_thr
        mm = 0.0
        for s, e in _find_clusters(suprap):
            mm = max(mm, _cluster_mass(tp, s, e))
        max_masses[p] = mm

    for c in clusters:
        c["p_value"] = float((np.sum(max_masses >= c["mass"]) + 1) / (n_perm + 1))

    return {"t_obs": t_obs, "t_thr": t_thr, "clusters": clusters, "df": n-1}


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------
def _format_p(p):
    return "p < 0.00001" if p < 1e-5 else f"p = {p:.4f}"


def add_phases_and_background(ax, show_phase_labels=False, purple_alpha=0.20):
    """Apply Stance/Swing backgrounds and draw subphase partitions."""
    ax.axvspan(0, STANCE_END, color="white", alpha=1.0, zorder=0)
    ax.axvspan(STANCE_END, 100, color="#FFD4A3", alpha=purple_alpha, zorder=0)
    add_phase_lines(ax, color="0.75", linestyle="--", linewidth=1.0, zorder=1)

    if show_phase_labels:
        for name, (a, b) in PHASES_8.items():
            xc = (a+b)/2
            ax.text(xc, 1.02, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=11)


def add_phase_lines(ax, color="0.75", linestyle="--", linewidth=1.0, zorder=1):
    """Draw vertical boundary lines separating the 8 Perry subphases."""
    boundaries = sorted(set([0, 100] + [v for rng in PHASES_8.values() for v in rng]))
    for b in boundaries:
        ax.axvline(b, color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder)


def shade_between_curve_and_threshold(ax, x, t, t_thr, start_idx, end_idx, alpha=0.25):
    """Shade the region between the observed t-curve and the critical threshold."""
    xs = x[start_idx:end_idx+1]
    ts = t[start_idx:end_idx+1]
    sign = 1.0 if np.nanmean(ts) >= 0 else -1.0
    y_thr = sign * t_thr
    ax.fill_between(xs, ts, y_thr, where=None, interpolate=True,
                    color="0.5", alpha=alpha, linewidth=0, zorder=2)


def place_cluster_p_label(ax, x, t, start_idx, end_idx, p_text, ypad=0.06):
    """Intelligently position the cluster p-value label above or below the curve."""
    xs = x[start_idx:end_idx+1]
    ts = t[start_idx:end_idx+1]
    xc = float(np.nanmean(xs))
    
    if np.nanmean(ts) >= 0:
        yc = float(np.nanmax(ts)) + (ax.get_ylim()[1]-ax.get_ylim()[0]) * ypad
        va = "bottom"
    else:
        yc = float(np.nanmin(ts)) - (ax.get_ylim()[1]-ax.get_ylim()[0]) * ypad
        va = "top"

    ax.text(xc, yc, p_text, ha="center", va=va, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=5)


def plot_three_panel(x, A, A_lab, B, B_lab, spm_res, title, ylab_wave="Angle (deg)",
                     out_path="out.png", formats=("png",), dpi=600, show_legend=True,
                     purple_alpha=0.40, put_xlabel_only_bottom=True):
    """Generate a high-impact 3-panel SPM results scientific figure."""
    fig = plt.figure(figsize=(10.5, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 2.4, 0.7], hspace=0.18)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)

    # ---- Panel 1: Trajectories ----
    add_phases_and_background(ax1, show_phase_labels=True, purple_alpha=purple_alpha)
    mA, sdA = np.nanmean(A, axis=0), np.nanstd(A, axis=0, ddof=1)
    mB, sdB = np.nanmean(B, axis=0), np.nanstd(B, axis=0, ddof=1)

    colA = "black" if "TD" in A_lab else "green"
    colB = "black" if "TD" in B_lab else "green"
    lsA  = "-" if ("Right" in A_lab or "R" in A_lab or "TD" in A_lab) else "--"
    lsB  = "-" if ("Right" in B_lab or "R" in B_lab or "TD" in B_lab) else "--"
    
    ax1.plot(x, mA, linewidth=3.2, label=A_lab, color=colA, linestyle=lsA)
    ax1.fill_between(x, mA-sdA, mA+sdA, color=colA, alpha=0.18, linewidth=0)
    ax1.plot(x, mB, linewidth=3.2, label=B_lab, color=colB, linestyle=lsB)
    ax1.fill_between(x, mB-sdB, mB+sdB, color=colB, alpha=0.18, linewidth=0)

    ax1.set_ylabel(ylab_wave)
    ax1.set_title(title, pad=28)
    if show_legend:
        ax1.legend(loc="lower right", bbox_to_anchor=(0.985, 0.02), ncol=2,
                   frameon=True, facecolor="white", edgecolor="none", framealpha=0.85,
                   fontsize=11, handlelength=2.8, columnspacing=1.2, borderpad=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelbottom=False)

    # ---- Panel 2: SPM{t} Curve ----
    add_phases_and_background(ax2, show_phase_labels=False, purple_alpha=purple_alpha)
    t = spm_res["t_obs"]
    t_thr = spm_res["t_thr"]

    ax2.plot(x, t, color="black", linewidth=3.0)
    ax2.axhline(0, color="0.4", linestyle=":", linewidth=1.5)
    ax2.axhline(+t_thr, color="0.35", linestyle="--", linewidth=1.5)
    ax2.axhline(-t_thr, color="0.35", linestyle="--", linewidth=1.5)

    ymax = max(np.nanmax(np.abs(t)), t_thr) * 1.15
    if not np.isfinite(ymax) or ymax == 0: ymax = 1.0
    ax2.set_ylim(-ymax, ymax)

    tx = x.max() - 1e-6
    ty = min(t_thr + 0.06*ymax, 0.98*ymax)
    ax2.text(tx, ty, f"t* = {t_thr:.3f}", ha="right", va="bottom", fontsize=11, color="crimson",
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85))

    sig_clusters = [c for c in spm_res["clusters"] if c["p_value"] < 0.05]
    for c in sig_clusters:
        shade_between_curve_and_threshold(ax2, x, t, t_thr, c["start"], c["end"], alpha=0.25)
        place_cluster_p_label(ax2, x, t, c["start"], c["end"], _format_p(c["p_value"]))

    ax2.set_ylabel("SPM{t}")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    if put_xlabel_only_bottom:
        ax2.tick_params(labelbottom=False)
    else:
        ax2.set_xlabel("Gait cycle (%)")

    # ---- Panel 3: Significance Horizon Bar ----
    add_phase_lines(ax3, color="0.75", linestyle="--", linewidth=1.0, zorder=5)
    ax3.set_ylim(0, 1.02)
    ax3.set_yticks([])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    ax3.add_patch(plt.Rectangle((0, 0.1), 100, 0.8, facecolor="white", edgecolor="black", linewidth=1.0))
    for c in sig_clusters:
        xs, xe = x[c["start"]], x[c["end"]]
        ax3.add_patch(plt.Rectangle((xs, 0.1), xe-xs, 0.8, facecolor="black", edgecolor="black", linewidth=0))
        ax3.text(0.5*(xs+xe), 0.902, _format_p(c["p_value"]), ha="center", va="bottom", fontsize=11, color="black")

    ax3.set_xlabel("Gait cycle (%)")
    ax3.set_xlim(0, 100)

    stem, _ = os.path.splitext(out_path)
    save_figure(fig, os.path.dirname(stem) or ".", os.path.basename(stem), formats=formats, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Spreadsheet Export Generators
# ---------------------------------------------------------------------------
def _cluster_direction(t_obs, start, end):
    mean_val = np.nanmean(np.asarray(t_obs[start:end+1], float))
    return "positive" if mean_val > 0 else "negative" if mean_val < 0 else "mixed"


def build_spm_timeseries_table(x, A, B, A_lab, B_lab, spm_res, joint, comparison, alpha):
    """Compile localized time-series statistics into a single Pandas DataFrame."""
    x = np.asarray(x, float)
    mA, sdA = np.nanmean(A, axis=0), np.nanstd(A, axis=0, ddof=1)
    mB, sdB = np.nanmean(B, axis=0), np.nanstd(B, axis=0, ddof=1)
    t = np.asarray(spm_res["t_obs"], float)
    t_thr = float(spm_res["t_thr"])
    supra = (np.abs(t) > t_thr).astype(int)

    cl_id = np.full_like(t, fill_value=-1, dtype=int)
    cl_p  = np.full_like(t, fill_value=np.nan, dtype=float)
    sig_clusters = [c for c in spm_res.get("clusters", []) if float(c.get("p_value", 1.0)) < 0.05]
    for i, c in enumerate(sig_clusters, start=1):
        s, e = int(c["start"]), int(c["end"])
        cl_id[s:e+1] = i
        cl_p[s:e+1]  = float(c["p_value"])

    return pd.DataFrame({
        "joint": joint, "comparison": comparison, "alpha": alpha, "gait_cycle_pct": x,
        "groupA": A_lab, "groupB": B_lab, "meanA": mA, "sdA": sdA, "meanB": mB, "sdB": sdB,
        "spm_t": t, "t_star": t_thr, "supra_threshold": supra, "sig_cluster_id": cl_id, "sig_cluster_p": cl_p,
    })


def build_spm_cluster_table(x, spm_res, joint, comparison, alpha, n_perm, seed, df_used):
    """Compile high-level topological cluster information into a DataFrame."""
    rows = []
    t = np.asarray(spm_res["t_obs"], float)
    t_thr = float(spm_res["t_thr"])
    for k, c in enumerate(spm_res.get("clusters", []), start=1):
        s, e = int(c["start"]), int(c["end"])
        rows.append({
            "joint": joint, "comparison": comparison, "alpha": alpha, "n_perm": n_perm, "seed": seed, "df": df_used,
            "cluster_id": k, "start_index": s, "end_index": e, "start_gc_pct": float(x[s]), "end_gc_pct": float(x[e]),
            "extent_points": int(e - s + 1), "extent_gc_pct": float(x[e] - x[s]), "t_star": t_thr,
            "mass": float(c.get("mass", np.nan)), "p_value": float(c.get("p_value", np.nan)),
            "direction": _cluster_direction(t, s, e), "significant_0p05": bool(float(c.get("p_value", 1.0)) < 0.05),
        })
    return pd.DataFrame(rows)


def export_spm_tables(outdir, timeseries_dfs, cluster_dfs, meta_rows, fname_stem="SPM_Results"):
    """Export analytical summary tables to Excel workbook and standalone CSVs."""
    os.makedirs(outdir, exist_ok=True)
    ts = pd.concat(timeseries_dfs, ignore_index=True) if timeseries_dfs else pd.DataFrame()
    cl = pd.concat(cluster_dfs, ignore_index=True) if cluster_dfs else pd.DataFrame()
    meta = pd.DataFrame(meta_rows)

    xlsx_path = os.path.join(outdir, f"{fname_stem}.xlsx")
    csv_ts_path = os.path.join(outdir, f"{fname_stem}_timeseries.csv")
    csv_cl_path = os.path.join(outdir, f"{fname_stem}_clusters.csv")

    if not ts.empty: ts.to_csv(csv_ts_path, index=False)
    if not cl.empty: cl.to_csv(csv_cl_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        meta.to_excel(writer, index=False, sheet_name="Meta")
        ts.to_excel(writer, index=False, sheet_name="Timeseries")
        cl.to_excel(writer, index=False, sheet_name="Clusters")

    return xlsx_path, csv_ts_path, csv_cl_path


# ---------------------------------------------------------------------------
# Execution Arguments Configuration Parser
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Gait 1D SPM Analysis Tool Suite")
    ap.add_argument("--outdir", type=str, default="OUT_Paper3_FINAL")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--analysis_level", type=str, default="trial", choices=["trial", "subject"])
    ap.add_argument("--purple_alpha", type=float, default=0.20)
    ap.add_argument("--plot_formats", type=str, default="png,pdf")
    ap.add_argument("--plot_dpi", type=int, default=1200)
    ap.add_argument("--font", type=str, default="Times New Roman")

    # File path arguments for joint kinematics
    ap.add_argument("--td_hip", required=True)
    ap.add_argument("--td_knee", required=True)
    ap.add_argument("--td_ankle", required=True)
    ap.add_argument("--cpR_hip", required=True)
    ap.add_argument("--cpR_knee", required=True)
    ap.add_argument("--cpR_ankle", required=True)
    ap.add_argument("--cpL_hip", required=True)
    ap.add_argument("--cpL_knee", required=True)
    ap.add_argument("--cpL_ankle", required=True)
    return ap.parse_args()


def load_joint(path, x):
    df = pd.read_excel(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    wcols = find_wave_cols(df)
    return df, wcols


def get_waves(df, wcols, x, level, group_kind):
    if level == "trial":
        return waves_trial_level(df, wcols, x)
    if group_kind == "TD":
        return waves_subject_level_TD(df, wcols, x)
    return waves_subject_level_CP(df, wcols, x)


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    
    # Resolving absolute paths safely
    args.td_hip = _norm_path(args.td_hip)
    args.td_knee = _norm_path(args.td_knee)
    args.td_ankle = _norm_path(args.td_ankle)
    args.cpR_hip = _norm_path(args.cpR_hip)
    args.cpR_knee = _norm_path(args.cpR_knee)
    args.cpR_ankle = _norm_path(args.cpR_ankle)
    args.cpL_hip = _norm_path(args.cpL_hip)
    args.cpL_knee = _norm_path(args.cpL_knee)
    args.cpL_ankle = _norm_path(args.cpL_ankle)
    
    set_plot_style(font=args.font, base_size=14)
    outdir = _norm_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    formats = tuple([s.strip().lower() for s in str(args.plot_formats).split(",") if s.strip()]) or ("png",)
    x = np.linspace(0, 100, 51) # 51 points sequence across 0-100%

    joints = [
        ("Hip",   args.td_hip,   args.cpR_hip,   args.cpL_hip),
        ("Knee",  args.td_knee,  args.cpR_knee,  args.cpL_knee),
        ("Ankle", args.td_ankle, args.cpR_ankle, args.cpL_ankle),
    ]

    timeseries_dfs, cluster_dfs, meta_rows = [], [], []

    for jname, td_path, cpr_path, cpl_path in joints:
        td_df, td_w = load_joint(td_path, x)
        cpr_df, cpr_w = load_joint(cpr_path, x)
        cpl_df, cpl_w = load_joint(cpl_path, x)

        TD  = get_waves(td_df, td_w, x, args.analysis_level, "TD")
        CPR = get_waves(cpr_df, cpr_w, x, args.analysis_level, "CP")
        CPL = get_waves(cpl_df, cpl_w, x, args.analysis_level, "CP")

        # Two-sample t-tests (CP vs TD)
        res_cpr = twosample_cluster_perm(CPR, TD, alpha=args.alpha, n_perm=args.n_perm, seed=args.seed)
        res_cpl = twosample_cluster_perm(CPL, TD, alpha=args.alpha, n_perm=args.n_perm, seed=args.seed)

        # Paired samples t-test (CP Right vs CP Left)
        n = min(len(CPR), len(CPL))
        res_rl = paired_cluster_perm(CPR[:n], CPL[:n], alpha=args.alpha, n_perm=args.n_perm, seed=args.seed)

        # Append structured metrics data rows 
        timeseries_dfs.append(build_spm_timeseries_table(x, TD, CPR, "TD (Right)", "CP-RIGHT", res_cpr, jname, "TD_vs_CP-RIGHT", args.alpha))
        cluster_dfs.append(build_spm_cluster_table(x, res_cpr, jname, "TD_vs_CP-RIGHT", args.alpha, args.n_perm, args.seed, res_cpr.get("df", np.nan)))
        meta_rows.append({"joint": jname, "comparison": "TD_vs_CP-RIGHT", "nA": int(TD.shape[0]), "nB": int(CPR.shape[0]), "analysis_level": args.analysis_level, "test": "two-sample", "t_star": float(res_cpr["t_thr"])})

        timeseries_dfs.append(build_spm_timeseries_table(x, TD, CPL, "TD (Right)", "CP-LEFT", res_cpl, jname, "TD_vs_CP-LEFT", args.alpha))
        cluster_dfs.append(build_spm_cluster_table(x, res_cpl, jname, "TD_vs_CP-LEFT", args.alpha, args.n_perm, args.seed, res_cpl.get("df", np.nan)))
        meta_rows.append({"joint": jname, "comparison": "TD_vs_CP-LEFT", "nA": int(TD.shape[0]), "nB": int(CPL.shape[0]), "analysis_level": args.analysis_level, "test": "two-sample", "t_star": float(res_cpl["t_thr"])})

        timeseries_dfs.append(build_spm_timeseries_table(x, CPR[:n], CPL[:n], "CP-RIGHT", "CP-LEFT", res_rl, jname, "CP-RIGHT_vs_CP-LEFT", args.alpha))
        cluster_dfs.append(build_spm_cluster_table(x, res_rl, jname, "CP-RIGHT_vs_CP-LEFT", args.alpha, args.n_perm, args.seed, res_rl.get("df", np.nan)))
        meta_rows.append({"joint": jname, "comparison": "CP-RIGHT_vs_CP-LEFT", "nA": int(CPR[:n].shape[0]), "nB": int(CPL[:n].shape[0]), "analysis_level": args.analysis_level, "test": "paired", "t_star": float(res_rl["t_thr"])})

        # Generate 3-panel figures
        plot_three_panel(x, TD, "TD (Right)", CPR, "CP-RIGHT", res_cpr, title=f"{jname} sagittal angle — TD(R) vs CP-RIGHT", out_path=os.path.join(outdir, f"TD_vs_CPRight_{jname}_paper3style.png"), formats=formats, dpi=args.plot_dpi, purple_alpha=args.purple_alpha)
        plot_three_panel(x, TD, "TD (Right)", CPL, "CP-LEFT", res_cpl, title=f"{jname} sagittal angle — TD(R) vs CP-LEFT", out_path=os.path.join(outdir, f"TD_vs_CPLeft_{jname}_paper3style.png"), formats=formats, dpi=args.plot_dpi, purple_alpha=args.purple_alpha)
        plot_three_panel(x, CPR, "CP-RIGHT", CPL, "CP-LEFT", res_rl, title=f"{jname} sagittal angle — CP-RIGHT vs CP-LEFT", out_path=os.path.join(outdir, f"CPRight_vs_CPLeft_{jname}_paper3style.png"), formats=formats, dpi=args.plot_dpi, purple_alpha=args.purple_alpha)

    # Export statistical findings to spreadsheet documents
    xlsx_path, _, _ = export_spm_tables(outdir, timeseries_dfs, cluster_dfs, meta_rows, fname_stem="SPM_Results")
    print("Execution completed successfully! Outputs saved inside directory:", outdir)
    print("Analytical Excel report generated:", xlsx_path)


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # FILE PATH PLACEHOLDERS FOR USERS
    # -----------------------------------------------------------------------
    # If the user executes the script directly without specifying arguments,
    # the script will look for these default file paths.
    if (len(sys.argv) == 1) or ("--td_hip" not in sys.argv):
        sys.argv = [
            sys.argv[0],
            "--outdir",      "OUT_Paper3_FINAL_v3",
            
            # --- PLACEHOLDERS: Replace these string literals with your file names ---
            "--td_hip",      "YOUR_DATA_FOLDER/Hip_TD_Data.xlsx",
            "--td_knee",     "YOUR_DATA_FOLDER/Knee_TD_Data.xlsx",
            "--td_ankle",    "YOUR_DATA_FOLDER/Ankle_TD_Data.xlsx",
            
            "--cpL_hip",     "YOUR_DATA_FOLDER/Hip_CP_Left_Data.xlsx",
            "--cpL_knee",    "YOUR_DATA_FOLDER/Knee_CP_Left_Data.xlsx",
            "--cpL_ankle",   "YOUR_DATA_FOLDER/Ankle_CP_Left_Data.xlsx",
            
            "--cpR_hip",     "YOUR_DATA_FOLDER/Hip_CP_Right_Data.xlsx",
            "--cpR_knee",    "YOUR_DATA_FOLDER/Knee_CP_Right_Data.xlsx",
            "--cpR_ankle",   "YOUR_DATA_FOLDER/Ankle_CP_Right_Data.xlsx",
            # -----------------------------------------------------------------------
            
            "--analysis_level", "trial",
            "--alpha",          "0.01",
            "--n_perm",         "5000",
            "--seed",           "42",
            "--purple_alpha",   "0.20",
        ]
    main()
