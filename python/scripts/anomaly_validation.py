"""Anomaly-form validation: LSME emissivity anomaly versus the TELSEM anomaly.

Reads the saved monthly-mean LSME emissivity composites (LSME_emis_F13_YYYYMM.nc,
written by monthly_lsme --save-emis), loads the matching TELSEM monthly atlases on
the same grid, removes each cell's mean over the available months from both, and
compares the resulting anomalies per channel over land.

Why this matters. The absolute monthly emissivity is dominated by the fixed
surface-type pattern (deserts emit high, forests lower), which is the same every
month and inflates the correlation against a climatology without measuring change.
The anomaly removes that static map and keeps the time-varying part, so the
anomaly skill is the honest measure of whether the derived emissivity tracks the
seasonal variation of the TELSEM climatology. The table reports both the absolute
pooled correlation and the anomaly correlation so the contrast is explicit.

Usage (wherever the composites and data/telsem2 live):
    python -m scripts.anomaly_validation --emis-dir ../scratch/lsme_monthly \\
        --telsem ../data/telsem2 --out ../scratch/anomaly [--min-days N] [--coarsen K]

--min-days N (default 1) applies a clear-day floor, where a cell's monthly mean is
used only when at least N clear days contributed that month. It thins the
thinnest-sampled swath edges and reads the n_obs stored in each monthly composite,
so no recompute is needed.

--coarsen K (default 1) block-averages the 0.25 degree fields to K times that grid
before the anomaly (K=2 gives 0.5 degree, K=4 gives 1.0 degree). Coarsening
averages across the orbital swath-sampling stripes, so it is the effective fix for
the arcs in the per-cell map.

--min-signal X (default 0) drops channel-cells where the TELSEM seasonal amplitude
(the per-cell standard deviation over the months) is below X. Those cells, mainly
hyper-arid deserts, have essentially no seasonal cycle, so the anomaly correlation
there is noise. X around 0.006 to 0.008 (emissivity units) masks the no-signal
desert cores. The three options compose.

Default (no holdout, no floor) is the screen-off baseline: the headline numbers
reported by the original Anomaly_Form_Analysis derivation. To use a no-signal
floor honestly, also pass:

--tune-months M[,M...] (default empty) months (1-12) used to compute the TELSEM
seasonal amplitude that the --min-signal floor is applied against. The skill
numbers are then reported on the held-out months only (months not in the tune
set, unless --report-months is given).

--report-months M[,M...] (default: complement of --tune-months) months used to
score skill. Defaults to whatever months are not in --tune-months when a tuning
split is given.

Without --tune-months a --min-signal > 0 run prints a leakage warning: the floor
is being chosen from the same months it is reported against.
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swi import anomaly, io_lsme_monthly, telsem, validate
from swi.channels import CHANNEL_NAMES, N_CHANNELS


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def discover(emis_dir, sat="F13"):
    """Chronological (year, month, path) of the saved monthly composites."""
    out = []
    for p in sorted(glob.glob(os.path.join(emis_dir, f"LSME_emis_{sat}_??????.nc"))):
        m = re.search(r"_(\d{4})(\d{2})\.nc$", p)
        out.append((int(m.group(1)), int(m.group(2)), p))
    return out


def _parse_month_csv(s):
    """Parse a comma-separated list of month numbers like '1,4,7,10' into a set."""
    if not s:
        return set()
    return {int(x) for x in s.split(",") if x.strip()}


def main(argv):
    emis_dir = opt(argv, "--emis-dir", "../scratch/lsme_monthly")
    sat = opt(argv, "--sat", "F13").upper()
    telsem_dir = opt(argv, "--telsem", "../data/telsem2")
    out = opt(argv, "--out", "../scratch/anomaly")
    min_days = int(opt(argv, "--min-days", "1"))
    coarsen = int(opt(argv, "--coarsen", "1"))
    min_signal = float(opt(argv, "--min-signal", "0"))
    tune_months = _parse_month_csv(opt(argv, "--tune-months", ""))
    report_months = _parse_month_csv(opt(argv, "--report-months", ""))
    os.makedirs(out, exist_ok=True)

    months = discover(emis_dir, sat)
    if len(months) < 2:
        print(f"need >= 2 monthly composites in {emis_dir}; found {len(months)}")
        return 1
    labels = [f"{y}-{m:02d}" for y, m, _ in months]
    print(f"months ({len(months)}): " + ", ".join(labels))

    # Tuning / reporting split. If --tune-months is given, the --min-signal floor
    # is chosen from TELSEM seasonal std on those months only, and the skill
    # numbers are reported on the report-months (default: months NOT in tune).
    # Without --tune-months the floor (if any) is in-sample on every month and
    # the run prints a clear note about the leakage.
    avail = [m for _, m, _ in months]
    if tune_months:
        if not report_months:
            report_months = set(avail) - tune_months
        tune_idx = np.array([m in tune_months for m in avail], dtype=bool)
        report_idx = np.array([m in report_months for m in avail], dtype=bool)
        if not report_idx.any():
            print("no months left for reporting after --tune-months and "
                  "--report-months; pick different sets")
            return 1
        if tune_idx.sum() < 2:
            # temporal_mean uses min_valid=2 by default, so a one-month tune
            # baseline is NaN everywhere and the anomaly metrics collapse to
            # NaN silently. Fail loud instead.
            print(f"--tune-months selects {int(tune_idx.sum())} month(s) but "
                  "the per-cell baseline needs at least 2 valid months to be "
                  "defined. Pick at least two tune months.")
            return 1
        print(f"holdout split: tune on {sorted(tune_months)}, report on "
              f"{sorted(m for m in avail if report_idx[avail.index(m)])}")
    else:
        tune_idx = np.ones(len(avail), dtype=bool)
        report_idx = tune_idx.copy()
        if min_signal > 0:
            print("WARN: --min-signal > 0 chosen on the same months it is "
                  "reported against (in-sample). Pass --tune-months to split "
                  "honestly.")

    lat = lon = None
    lsme_list = []
    for _, _, p in months:
        lat, lon, emis, nobs = io_lsme_monthly.read_monthly_emis(p)
        if min_days > 1:
            # Clear-day floor: drop a cell's monthly mean where fewer than min_days
            # clear days contributed. This thins the thinly-sampled swath-edge cells
            # that stripe the per-cell anomaly map without changing the well-sampled
            # interior.
            emis = np.where(nobs >= min_days, emis, np.nan)
        lsme_list.append(emis)
    lsme_stack = np.stack(lsme_list, axis=0)              # (T, nlat, nlon, 7)
    if min_days > 1:
        print(f"clear-day floor: >= {min_days} clear days per cell-month")

    telsem_list = []
    for _, mo, _ in months:
        _, _, atlas = telsem.load_atlas(telsem_dir, mo, lat=lat, lon=lon)
        telsem_list.append(atlas)
    telsem_stack = np.stack(telsem_list, axis=0)

    land = telsem.land_mask(lat, lon)

    if coarsen > 1:
        # Coarsen the monthly fields to a larger grid before the anomaly, which
        # averages across the orbital swath-sampling stripes.
        for ax in (1, 2):
            lsme_stack = anomaly.block_mean(lsme_stack, coarsen, ax)
            telsem_stack = anomaly.block_mean(telsem_stack, coarsen, ax)
        lat = anomaly.block_mean(lat, coarsen, 0)
        lon = anomaly.block_mean(lon, coarsen, 0)
        land = (anomaly.block_mean(anomaly.block_mean(land.astype(float), coarsen, 0),
                                   coarsen, 1) >= 0.5)
        print(f"coarsened {coarsen}x to {0.25 * coarsen:g} deg: "
              f"grid now {lat.size} x {lon.size}")

    if min_signal > 0:
        # No-signal mask. Where the TELSEM seasonal amplitude (the per-cell standard
        # deviation over the months) is below the floor, there is essentially no
        # seasonal cycle to validate against, so the per-cell anomaly correlation is
        # noise (often negative). Drop those channel-cells, which are the hyper-arid
        # deserts. The gate is on TELSEM because it is the trustworthy reference
        # amplitude; the derived field's noise can carry spurious variance.
        # Compute the floor on TUNE months only so the same TELSEM data is not
        # both selecting the cells and being scored against, then APPLY the
        # resulting cell mask to the full stack.
        with np.errstate(invalid="ignore"):
            tsig = np.nanstd(telsem_stack[tune_idx], axis=0)
        low = tsig < min_signal
        lsme_stack = np.where(low[None], np.nan, lsme_stack)
        telsem_stack = np.where(low[None], np.nan, telsem_stack)
        scope = "all months" if tune_idx.all() else f"tune months only ({int(tune_idx.sum())}/{len(avail)})"
        print(f"signal floor: TELSEM seasonal std >= {min_signal:g} "
              f"({scope}); masked {100 * float(low.mean()):.1f}% of channel-cells as no-signal")

    # Per-cell anomaly baseline is computed on TUNE months only and subtracted
    # from the full stack, so the per-cell mean used to define each report
    # month's anomaly is not informed by the report months themselves. When no
    # tuning split is given, this reduces to the original full-stack baseline.
    lsme_baseline = anomaly.temporal_mean(lsme_stack[tune_idx])
    telsem_baseline = anomaly.temporal_mean(telsem_stack[tune_idx])
    lsme_anom = lsme_stack - lsme_baseline
    telsem_anom = telsem_stack - telsem_baseline

    # Restrict scoring to the reporting months. The baseline is fixed (tune set
    # only), so the report-month anomalies are honestly held out when a split
    # is active. Skill metrics are computed on the report-month slice.
    if not report_idx.all():
        lsme_anom = lsme_anom[report_idx]
        telsem_anom = telsem_anom[report_idx]
        lsme_stack_report = lsme_stack[report_idx]
        telsem_stack_report = telsem_stack[report_idx]
        report_labels = [labels[i] for i in range(len(labels)) if report_idx[i]]
        print(f"scoring on heldout months: {', '.join(report_labels)}")
    else:
        lsme_stack_report = lsme_stack
        telsem_stack_report = telsem_stack

    rows = []
    for c in range(N_CHANNELS):
        # pooled absolute skill over land (dominated by the static spatial
        # pattern), then the anomaly skill on the same cells. Both are computed
        # on the report months only when a holdout split is in effect.
        _, abs_agg = anomaly.anomaly_pattern_skill(
            lsme_stack_report[..., c], telsem_stack_report[..., c], mask=land)
        per, an_agg = anomaly.anomaly_pattern_skill(
            lsme_anom[..., c], telsem_anom[..., c], mask=land)
        rows.append({"ch": CHANNEL_NAMES[c], "abs_r": abs_agg["pearson_r"],
                     "anom_r": an_agg["pearson_r"], "anom_rho": an_agg["spearman_r"],
                     "anom_rmse": an_agg["rmse"], "n": an_agg["n"], "per": per})

    print(f"\n{'ch':>4} {'n':>9} {'abs_r':>7} {'anom_r':>7} {'anom_rho':>9} "
          f"{'anom_rmse':>10}")
    for r in rows:
        print(f"{r['ch']:>4} {r['n']:>9,} {r['abs_r']:>7.3f} {r['anom_r']:>7.3f} "
              f"{r['anom_rho']:>9.3f} {r['anom_rmse']:>10.4f}")

    # per-cell temporal anomaly correlation map for a strong window channel (37H),
    # when there are enough months to define it per cell. Computed on the report
    # months when a holdout split is in effect.
    c37h = CHANNEL_NAMES.index("37H")
    min_n = max(2, lsme_stack_report.shape[0] // 2)
    rmap, nmap = validate.temporal_anomaly_correlation(
        lsme_stack_report[..., c37h], telsem_stack_report[..., c37h], min_n=min_n)
    rmap = np.where(land, rmap, np.nan)
    med = float(np.nanmedian(rmap)) if np.isfinite(rmap).any() else float("nan")
    print(f"\nper-cell temporal anomaly r (37H, min_n={min_n}): "
          f"median {med:.3f} over {int(np.isfinite(rmap).sum()):,} land cells")

    fig_summary(rows, lat, lon, rmap, lsme_anom, telsem_anom, land, labels, out,
                min_days, coarsen, min_signal, sat)
    return 0


def roll180(a, lon):
    return np.roll(a, lon.size // 2, axis=1)


def fig_summary(rows, lat, lon, rmap, lsme_anom, telsem_anom, land, labels, out,
                min_days=1, coarsen=1, min_signal=0.0, sat="F13"):
    floor = "" if min_days <= 1 else f", clear-day floor >= {min_days}/cell-month"
    grid = "" if coarsen <= 1 else f", {0.25 * coarsen:g} deg grid"
    sig = "" if min_signal <= 0 else f", signal floor {min_signal:g}"
    sat_disp = f"{sat[:1]}-{sat[1:]}"
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"LSME emissivity anomaly vs TELSEM anomaly, {sat_disp} 1998 "
                 f"({len(labels)} months: {labels[0]}..{labels[-1]}{floor}{grid}{sig})",
                 fontsize=13)

    # (1) absolute vs anomaly correlation per channel
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(rows))
    ax1.bar(x - 0.2, [r["abs_r"] for r in rows], 0.4, label="absolute (static pattern)",
            color="#bdbdbd")
    ax1.bar(x + 0.2, [r["anom_r"] for r in rows], 0.4, label="anomaly (time-varying)",
            color="#1f77b4")
    ax1.set_xticks(x); ax1.set_xticklabels([r["ch"] for r in rows])
    ax1.set_ylabel("Pearson correlation"); ax1.set_ylim(0, 1)
    ax1.set_title("absolute vs anomaly skill per channel", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # (2) pooled anomaly density for 19H (a strong window channel)
    ax2 = fig.add_subplot(2, 2, 2)
    c19h = CHANNEL_NAMES.index("19H")
    a = lsme_anom[..., c19h]; b = telsem_anom[..., c19h]
    m = np.isfinite(a) & np.isfinite(b) & land[None, :, :]
    av, bv = a[m], b[m]
    lim = np.nanpercentile(np.abs(np.concatenate([av, bv])), 99) if av.size else 0.05
    ax2.hist2d(bv, av, bins=80, range=[[-lim, lim], [-lim, lim]], cmap="viridis",
               cmin=1)
    ax2.plot([-lim, lim], [-lim, lim], "r-", lw=0.8)
    rr = np.corrcoef(av, bv)[0, 1] if av.size > 3 else float("nan")
    ax2.set_title(f"19H pooled anomaly (r = {rr:.3f}, n = {av.size:,})", fontsize=10)
    ax2.set_xlabel("TELSEM anomaly"); ax2.set_ylabel("LSME anomaly")

    # (3) per-cell temporal anomaly correlation map (37H)
    ax3 = fig.add_subplot(2, 1, 2)
    ext = (-180, 180, lat[0], lat[-1])
    im = ax3.imshow(roll180(rmap, lon), origin="lower", extent=ext, cmap="RdBu_r",
                    vmin=-1, vmax=1, aspect="auto")
    ax3.set_title("per-cell temporal anomaly correlation, 37H (LSME vs TELSEM)",
                  fontsize=10)
    ax3.set_xlabel("longitude"); ax3.set_ylabel("latitude")
    fig.colorbar(im, ax=ax3, shrink=0.8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = ("" if min_days <= 1 else f"_min{min_days}") + \
             ("" if coarsen <= 1 else f"_c{coarsen}") + \
             ("" if min_signal <= 0 else f"_s{min_signal:g}")
    p = os.path.join(out, f"anomaly_validation_{sat}{suffix}.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
