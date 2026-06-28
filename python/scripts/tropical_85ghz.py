"""Why does the 85 GHz emissivity degrade in the tropics? (F-13 July 1998 vs TELSEM)

The 31-day validation showed 85V pattern correlation falling toward the equator
(0.68 tropics, 0.80 midlat, 0.87 high-lat) while the window channels stay high
everywhere. This script tests the competing explanations directly:

  H1  residual water vapour: stratify the 85V (and 85H, 22V) residual against
      TELSEM by ERA5 column water vapour. If bias drifts negative and RMSE grows
      with tcwv, undercorrected/residual atmosphere is implicated.
  H2  correctable vs not: recompute 85V and 22V with the ERA5 first-order
      atmospheric correction ON and OFF (nominal), same GridSat Ts and domain.
      Whether the correction helps, overcorrects, or barely moves the tropics
      separates correctable atmosphere from scattering/signal limits.
  H3  dynamic range: compare the spatial spread of TELSEM and derived 85 GHz
      emissivity by zone. A compressed tropical range means low correlation is
      partly a low-signal problem, not pure error.

Writes a stats JSON and figures (tcwv stratification, a tcwv-residual density,
and the correction-efficacy bars) to --out.

Usage (in-env):
    python -m scripts.tropical_85ghz \
        --gridsat-root /path/to/swi_data/data/gridsat_cld --out ../scratch/tropics85
"""

import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swi import io_csu_grid, lsme, telsem, geostationary
from swi.channels import CHANNEL_NAMES
from scripts.run_lsme import gridsat_ts, era5_field
from swi import validate

F13DIR_DEFAULT = "../data/f13_1998"
F13_FMT = "CSU_SSMI_FCDR-GRID_V02R00_F13_D{ym}{dd}.nc"
TCWV_BINS = [0, 15, 25, 35, 45, 55, 100]
VZA_BINS = [0, 30, 45, 55, 65, 81]                 # geostationary viewing zenith angle
VZA_THRESHOLDS = [30, 40, 45, 50, 55, 60]          # split points to sweep (Ken's refinement)
LAT_BANDS = [(lo, lo + 10) for lo in range(-60, 60, 10)]   # signed 10-deg bands
CH = {"22V": 2, "37H": 4, "85V": 5, "85H": 6}      # Basist channel indices


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def fmt(s):
    """Compact rmse/bias/n string for a side-of-threshold stats dict."""
    if s.get("rmse") is None:
        return f"n{s['n']:,}"
    return f"rmse{s['rmse']:.3f},bias{s['bias']:+.3f},n{s['n']:,}"


def discover_days(f13dir, ym):
    paths = glob.glob(os.path.join(f13dir, F13_FMT.format(ym=ym, dd="??")))
    return sorted(re.search(r"D\d{6}(\d\d)\.nc$", p).group(1) for p in paths)


def composite(f13dir, gridsat_root, days, ym, pass_="dsc"):
    """Month means: emissivity with ERA5 atmosphere and with nominal (no atmos),
    plus the mean morning ERA5 tcwv per cell. Same GridSat Ts and clear+land domain."""
    sa = ca = sn = cn = st = ct = lat = lon = land = None
    used = []
    for dd in days:
        f13 = os.path.join(f13dir, F13_FMT.format(ym=ym, dd=dd))
        if not os.path.exists(f13):
            continue
        lat, lon, tb, _ = io_csu_grid.read_channels(f13, pass_=pass_)
        g = gridsat_ts(gridsat_root, lat, lon, csu_path=f13)
        if g is None:
            continue
        ts, clear_gs = g
        land = telsem.land_mask(lat, lon)
        clear = clear_gs & land
        tcwv = era5_field(f13, "tcwv", lat, lon)
        t_atm = era5_field(f13, "t2m", lat, lon)
        e_a = lsme.derive_emissivity(tb, ts, clear=clear, tcwv_mm=tcwv, t_atm=t_atm)["emissivity"]
        e_n = lsme.derive_emissivity(tb, ts, clear=clear, tcwv_mm=None, t_atm=None)["emissivity"]
        if sa is None:
            sa = np.zeros_like(e_a); ca = np.zeros_like(e_a)
            sn = np.zeros_like(e_a); cn = np.zeros_like(e_a)
            st = np.zeros(e_a.shape[:2]); ct = np.zeros(e_a.shape[:2])
        va = np.isfinite(e_a); sa[va] += e_a[va]; ca[va] += 1.0
        vn = np.isfinite(e_n); sn[vn] += e_n[vn]; cn[vn] += 1.0
        if tcwv is not None:
            vt = np.isfinite(tcwv); st[vt] += tcwv[vt]; ct[vt] += 1.0
        used.append(dd)
        print(f"07-{dd} ok")
    e_atm = np.where(ca > 0, sa / np.where(ca > 0, ca, 1.0), np.nan)
    e_nom = np.where(cn > 0, sn / np.where(cn > 0, cn, 1.0), np.nan)
    tcwv_m = np.where(ct > 0, st / np.where(ct > 0, ct, 1.0), np.nan)
    print(f"composite of {len(used)} days")
    return lat, lon, e_atm, e_nom, tcwv_m, land


def skill(a, b, m):
    sel = m & np.isfinite(a) & np.isfinite(b)
    if sel.sum() < 50:
        return {"n": int(sel.sum()), "pearson": None, "bias": None, "rmse": None}
    x, y = a[sel], b[sel]
    return {"n": int(sel.sum()),
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            "bias": float(np.mean(x - y)),
            "rmse": float(np.sqrt(np.mean((x - y) ** 2)))}


def main(argv):
    f13dir = opt(argv, "--f13-dir", F13DIR_DEFAULT)
    gridsat_root = opt(argv, "--gridsat-root", "../data/gridsat_july")
    out = opt(argv, "--out", "../scratch/tropics85")
    year = int(opt(argv, "--year", "1998"))
    month = int(opt(argv, "--month", "7"))
    ym = f"{year}{month:02d}"
    days = discover_days(f13dir, ym)
    os.makedirs(out, exist_ok=True)

    lat, lon, e_atm, e_nom, tcwv, land = composite(f13dir, gridsat_root, days, ym)
    _, _, atlas = telsem.load_atlas(None, month, lat=lat, lon=lon)

    lat2d = np.repeat(np.asarray(lat)[:, None], lon.size, axis=1)
    tropics = (np.abs(lat2d) < 23.5) & land
    midlat = (np.abs(lat2d) >= 23.5) & (np.abs(lat2d) < 60) & land
    highlat = (np.abs(lat2d) >= 60) & land
    zones = {"tropics": tropics, "midlat": midlat, "high-lat": highlat}

    summary = {}

    # H1/H3: zone stats + dynamic range for 85V, 85H, 22V, 37H
    print("\n=== Per-zone skill (ERA5 atmosphere) and TELSEM/derived spread ===")
    zone_stats = {}
    for name, ic in CH.items():
        a = e_atm[:, :, ic]; ref = atlas[:, :, ic]
        zs = {}
        for zn, zm in zones.items():
            s = skill(a, ref, zm)
            sel = zm & np.isfinite(a) & np.isfinite(ref)
            s["std_telsem"] = float(np.std(ref[sel])) if sel.sum() > 50 else None
            s["std_derived"] = float(np.std(a[sel])) if sel.sum() > 50 else None
            zs[zn] = s
        zone_stats[name] = zs
        print(f"{name}:")
        for zn, s in zs.items():
            if s["pearson"] is None:                 # zone has <50 valid cells
                print(f"   {zn:9s} n={s['n']:,} (no data)")
                continue
            # std_telsem/std_derived can be None one cell sooner than pearson
            # (skill uses <50, the std guard uses >50), so format them defensively.
            st = "n/a" if s["std_telsem"] is None else f"{s['std_telsem']:.3f}"
            sd = "n/a" if s["std_derived"] is None else f"{s['std_derived']:.3f}"
            print(f"   {zn:9s} r={s['pearson']:.3f} bias={s['bias']:+.3f} "
                  f"rmse={s['rmse']:.3f}  std(telsem)={st} std(derived)={sd}  "
                  f"n={s['n']:,}")
    summary["zone_stats"] = zone_stats

    # H1: residual vs tcwv, global and tropics, for 85V/85H/22V
    print("\n=== 85V/85H/22V residual stratified by ERA5 column water vapour ===")
    strat = {}
    for name, ic in {"85V": 5, "85H": 6, "22V": 2}.items():
        a = e_atm[:, :, ic]; ref = atlas[:, :, ic]
        res = a - ref
        rows = []
        for lo, hi in zip(TCWV_BINS[:-1], TCWV_BINS[1:]):
            m = land & np.isfinite(res) & np.isfinite(tcwv) & (tcwv >= lo) & (tcwv < hi)
            if m.sum() < 50:
                rows.append({"bin": f"{lo}-{hi}", "n": int(m.sum())}); continue
            x = a[m]; y = ref[m]
            rows.append({"bin": f"{lo}-{hi}", "n": int(m.sum()),
                         "tcwv_mid": (lo + hi) / 2,
                         "bias": float(np.mean(x - y)),
                         "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                         "pearson": float(np.corrcoef(x, y)[0, 1])})
        strat[name] = rows
        print(f"{name} by tcwv (mm): " + " | ".join(
            f"{r['bin']}:bias{r.get('bias', float('nan')):+.3f},rmse{r.get('rmse', float('nan')):.3f},n{r['n']:,}"
            for r in rows))
    summary["tcwv_strat"] = strat

    # H_VZA (Ken's test): 85V residual stratified by geostationary viewing zenith
    # angle. If the error FALLS at high VZA, the GridSat cloud mask over-flags there
    # (catching more cirrus) and thin cirrus near nadir is contaminating 85 GHz.
    vza = geostationary.min_geo_vza(lat, lon)
    ref85 = atlas[:, :, 5]
    a85 = e_atm[:, :, 5]
    vstrat = {}
    print("\n=== 85V residual stratified by geostationary VZA (Ken's test) ===")
    for zlabel, zmask in (("global", land), ("tropics", tropics)):
        rows = []
        for lo, hi in zip(VZA_BINS[:-1], VZA_BINS[1:]):
            m = zmask & np.isfinite(a85) & np.isfinite(ref85) & np.isfinite(vza) \
                & (vza >= lo) & (vza < hi)
            if m.sum() < 50:
                rows.append({"bin": f"{lo}-{hi}", "n": int(m.sum())}); continue
            x = a85[m]; y = ref85[m]
            rows.append({"bin": f"{lo}-{hi}", "n": int(m.sum()), "vza_mid": (lo + hi) / 2,
                         "bias": float(np.mean(x - y)),
                         "rmse": float(np.sqrt(np.mean((x - y) ** 2)))})
        vstrat[zlabel] = rows
        print(f"  85V {zlabel}: " + " | ".join(
            f"{r['bin']}:rmse{r.get('rmse', float('nan')):.3f},bias{r.get('bias', float('nan')):+.3f},n{r['n']:,}"
            for r in rows))
    summary["vza_strat"] = vstrat

    # Ken's refinement: sweep the split threshold to LOCATE where the error changes,
    # rather than relying on fixed bins. For each threshold, compare 85V error below
    # vs above it; error lower above the threshold supports the high-VZA over-flag idea.
    def side_stats(mask):
        m = mask & np.isfinite(a85) & np.isfinite(ref85) & np.isfinite(vza)
        if m.sum() < 50:
            return {"n": int(m.sum()), "bias": None, "rmse": None}
        x = a85[m]; y = ref85[m]
        return {"n": int(m.sum()), "bias": float(np.mean(x - y)),
                "rmse": float(np.sqrt(np.mean((x - y) ** 2)))}

    sweep = {}
    print("\n=== 85V error below vs above a VZA threshold (Ken: where does it change?) ===")
    for zlabel, zmask in (("global", land), ("tropics", tropics)):
        rows = []
        for thr in VZA_THRESHOLDS:
            rows.append({"thr": thr, "low": side_stats(zmask & (vza < thr)),
                         "high": side_stats(zmask & (vza >= thr))})
        sweep[zlabel] = rows
        print(f"  {zlabel}: " + " | ".join(
            f"thr{r['thr']}: lo {fmt(r['low'])} hi {fmt(r['high'])}" for r in rows))
    summary["vza_sweep"] = sweep

    # Confound-controlled test: within each latitude band, split by the band-median
    # VZA and compare 85V bias. This holds latitude (hence the snow/season signal)
    # roughly fixed, so a positive dbias = bias(high VZA) - bias(low VZA) across bands
    # is the VZA effect itself (Ken's cirrus removal), not the VZA-latitude confound.
    res85 = a85 - ref85
    lat2d = np.repeat(np.asarray(lat)[:, None], lon.size, axis=1)
    band_rows = []
    print("\n=== Within-latitude-band VZA decoupling (confound-controlled, Ken) ===")
    print("  dbias = bias(high VZA) - bias(low VZA) at fixed latitude; "
          "POSITIVE = high VZA cleaner = supports the cirrus story")
    for blo, bhi in LAT_BANDS:
        bm = land & (lat2d >= blo) & (lat2d < bhi) & np.isfinite(res85) & np.isfinite(vza)
        if bm.sum() < 400:
            band_rows.append({"band": [blo, bhi], "n": int(bm.sum())}); continue
        vmed = float(np.median(vza[bm]))
        lo_m = bm & (vza < vmed); hi_m = bm & (vza >= vmed)
        bl = float(np.mean(res85[lo_m])); bh = float(np.mean(res85[hi_m]))
        band_rows.append({"band": [blo, bhi], "vza_med": vmed, "bias_lowvza": bl,
                          "bias_highvza": bh, "dbias": bh - bl,
                          "n_lo": int(lo_m.sum()), "n_hi": int(hi_m.sum())})
        print(f"  lat[{blo:+03d},{bhi:+03d}) vzaMed={vmed:4.0f}  "
              f"bias lo={bl:+.3f} hi={bh:+.3f}  dbias={bh-bl:+.3f}  "
              f"n={lo_m.sum()+hi_m.sum():,}")
    summary["vza_latband"] = band_rows
    good = [r for r in band_rows if "dbias" in r]
    if good:
        wmean = sum(r["dbias"] * (r["n_lo"] + r["n_hi"]) for r in good) \
            / sum(r["n_lo"] + r["n_hi"] for r in good)
        pos = sum(1 for r in good if r["dbias"] > 0)
        summary["vza_latband_summary"] = {"weighted_mean_dbias": wmean,
                                          "bands_positive": pos, "bands_total": len(good)}
        print(f"  --> n-weighted mean dbias = {wmean:+.4f}; "
              f"{pos}/{len(good)} bands positive "
              f"({'supports Ken' if wmean > 0 else 'against Ken'})")
    # Dump the arrays so the band analysis can be iterated locally without re-running.
    np.savez(os.path.join(out, "vza_arrays.npz"), lat=np.asarray(lat),
             lon=np.asarray(lon), vza=vza, res85=res85, land=land)

    # H2: correction efficacy (ERA5 atmosphere vs nominal), 85V and 22V
    print("\n=== Atmospheric correction efficacy: ERA5-atmos vs nominal ===")
    eff = {}
    for name, ic in {"85V": 5, "22V": 2}.items():
        ref = atlas[:, :, ic]
        e = {}
        for zn, zm in {"global": land, **zones}.items():
            e[zn] = {"atmos": skill(e_atm[:, :, ic], ref, zm),
                     "nominal": skill(e_nom[:, :, ic], ref, zm)}
        eff[name] = e
        for zn in ("global", "tropics"):
            sa = e[zn]["atmos"]; sn = e[zn]["nominal"]
            print(f"{name} {zn:8s}: atmos r={sa['pearson']:.3f} rmse={sa['rmse']:.3f} "
                  f"bias={sa['bias']:+.3f}  | nominal r={sn['pearson']:.3f} "
                  f"rmse={sn['rmse']:.3f} bias={sn['bias']:+.3f}")
    summary["correction_efficacy"] = eff

    # --- figures ---
    fig_tcwv_strat(strat, out)
    fig_tcwv_density(e_atm, atlas, tcwv, tropics, out)
    fig_correction(eff, out)
    fig_vza(vstrat, out)
    fig_vza_threshold(sweep, out)
    fig_vza_latband(band_rows, out)

    jp = os.path.join(out, "tropics85_stats.json")
    with open(jp, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\nwrote", jp)
    return 0


def fig_tcwv_strat(strat, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("85 GHz and 22V emissivity error vs ERA5 column water vapour "
                 "(F-13 July 1998, land, vs TELSEM)", fontsize=12)
    colors = {"85V": "#d95f02", "85H": "#e7a", "22V": "#1b9e77"}
    for name, rows in strat.items():
        xs = [r["tcwv_mid"] for r in rows if "bias" in r]
        bias = [r["bias"] for r in rows if "bias" in r]
        rmse = [r["rmse"] for r in rows if "bias" in r]
        c = colors.get(name, None)
        a1.plot(xs, bias, "o-", color=c, label=name)
        a2.plot(xs, rmse, "o-", color=c, label=name)
    a1.axhline(0, color="k", lw=0.8, ls=":")
    a1.set_xlabel("column water vapour (mm)"); a1.set_ylabel("bias (derived - TELSEM)")
    a1.set_title("Bias drifts negative as the atmosphere wets up"); a1.legend(); a1.grid(alpha=0.3)
    a2.set_xlabel("column water vapour (mm)"); a2.set_ylabel("RMSE (emissivity)")
    a2.set_title("Scatter grows with water vapour"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "tcwv_stratification.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def fig_tcwv_density(e_atm, atlas, tcwv, tropics, out):
    res = e_atm[:, :, 5] - atlas[:, :, 5]
    m = tropics & np.isfinite(res) & np.isfinite(tcwv)
    x = tcwv[m]; y = res[m]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    hb = ax.hexbin(x, y, gridsize=50, bins="log", cmap="magma",
                   extent=(10, 70, -0.2, 0.15))
    ax.axhline(0, color="w", lw=1, ls=":")
    ax.set_xlabel("ERA5 column water vapour (mm)")
    ax.set_ylabel("85V residual (derived - TELSEM)")
    ax.set_title("Tropical land: 85V residual vs water vapour\n"
                 "(one-sided negative tail in humid air = scattering near convection)",
                 fontsize=11)
    fig.colorbar(hb, ax=ax, label="log10 count")
    fig.tight_layout()
    p = os.path.join(out, "tcwv_residual_density.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def fig_correction(eff, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Atmospheric-correction efficacy: ERA5 first-order vs none (nominal)",
                 fontsize=12)
    for ax, name in zip(axes, ("85V", "22V")):
        zones = ["global", "tropics", "midlat", "high-lat"]
        # empty zones (tropical-only input) have None Pearson; NaN bars are skipped
        ratm = [v if (v := eff[name][z]["atmos"]["pearson"]) is not None else np.nan
                for z in zones]
        rnom = [v if (v := eff[name][z]["nominal"]["pearson"]) is not None else np.nan
                for z in zones]
        x = np.arange(len(zones)); w = 0.38
        ax.bar(x - w/2, rnom, w, label="no correction", color="#bbbbbb")
        ax.bar(x + w/2, ratm, w, label="ERA5 correction", color="#2c7fb8")
        ax.set_xticks(x); ax.set_xticklabels(zones); ax.set_ylim(0, 1.0)
        ax.set_ylabel("Pearson r vs TELSEM"); ax.set_title(name)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "correction_efficacy.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def fig_vza_latband(band_rows, out):
    rows = [r for r in band_rows if "dbias" in r]
    if not rows:
        return
    cen = [(r["band"][0] + r["band"][1]) / 2 for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("VZA effect controlled for latitude: within each 10-deg band, "
                 "split at the band-median VZA", fontsize=11)
    a1.plot(cen, [r["bias_lowvza"] for r in rows], color="#2c7fb8", marker="o",
            linestyle="-", label="low VZA (below band median)")
    a1.plot(cen, [r["bias_highvza"] for r in rows], color="#d95f02", marker="s",
            linestyle="--", markerfacecolor="white", dashes=(5, 2),
            label="high VZA (above band median)")
    a1.axhline(0, color="k", lw=0.8, ls=":")
    a1.set_xlabel("latitude band center (deg)"); a1.set_ylabel("85V bias (derived - TELSEM)")
    a1.set_title("85V bias by latitude, low vs high VZA"); a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.bar(cen, [r["dbias"] for r in rows], width=8,
           color=["#4daf4a" if r["dbias"] > 0 else "#984ea3" for r in rows])
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xlabel("latitude band center (deg)")
    a2.set_ylabel("dbias = bias(high VZA) - bias(low VZA)")
    a2.set_title("Positive (green) = high VZA cleaner at fixed latitude\n"
                 "= supports Ken, free of the latitude confound", fontsize=9)
    a2.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "vza_latband.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def fig_vza_threshold(sweep, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("85V error split below (filled circle, solid) vs above "
                 "(open square, dashed) the VZA threshold -- Ken's refinement", fontsize=11)
    # Distinguish the two sides by MARKER SHAPE (circle vs square) as well as line
    # style, so the legend is readable even though its line samples are short.
    LO = dict(marker="o", linestyle="-")
    HI = dict(marker="s", linestyle="--", markerfacecolor="white", dashes=(5, 2))
    for zlabel, color in (("global", "#888888"), ("tropics", "#d95f02")):
        lo = [r for r in sweep.get(zlabel, []) if r["low"].get("rmse") is not None]
        hi = [r for r in sweep.get(zlabel, []) if r["high"].get("rmse") is not None]
        a1.plot([r["thr"] for r in lo], [r["low"]["bias"] for r in lo],
                color=color, label=f"{zlabel}: VZA below threshold (kept)", **LO)
        a1.plot([r["thr"] for r in hi], [r["high"]["bias"] for r in hi],
                color=color, label=f"{zlabel}: VZA above threshold", **HI)
        a2.plot([r["thr"] for r in lo], [r["low"]["rmse"] for r in lo], color=color, **LO)
        a2.plot([r["thr"] for r in hi], [r["high"]["rmse"] for r in hi], color=color, **HI)
    a1.axhline(0, color="k", lw=0.8, ls=":")
    a1.set_xlabel("VZA split threshold (deg)"); a1.set_ylabel("85V bias (derived - TELSEM)")
    a1.set_title("Bias: low-VZA side stays negative (scattering),\nhigh-VZA side near zero "
                 "= supports the cirrus story", fontsize=9)
    a1.legend(fontsize=7)
    a2.set_xlabel("VZA split threshold (deg)"); a2.set_ylabel("85V RMSE (emissivity)")
    a2.set_title("RMSE (rises at high VZA from oblique-view noise,\na confound to read with care)",
                 fontsize=9)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "vza_threshold_sweep.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def fig_vza(vstrat, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("85V emissivity error vs geostationary viewing zenith angle "
                 "(Ken's test: a drop at high VZA = the mask catches more cirrus there)",
                 fontsize=11)
    for zlabel, color in (("global", "#888888"), ("tropics", "#d95f02")):
        rows = [r for r in vstrat.get(zlabel, []) if "rmse" in r]
        xs = [r["vza_mid"] for r in rows]
        a1.plot(xs, [r["rmse"] for r in rows], "o-", color=color, label=zlabel)
        a2.plot(xs, [r["bias"] for r in rows], "o-", color=color, label=zlabel)
    a1.set_xlabel("geostationary VZA (deg)"); a1.set_ylabel("85V RMSE (emissivity)")
    a1.set_title("RMSE vs VZA"); a1.legend(); a1.grid(alpha=0.3)
    a2.axhline(0, color="k", lw=0.8, ls=":")
    a2.set_xlabel("geostationary VZA (deg)"); a2.set_ylabel("85V bias (derived - TELSEM)")
    a2.set_title("Bias vs VZA"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "vza_stratification.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
