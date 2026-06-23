"""Land Surface Microwave Emissivity (LSME), first-order estimator.

Steps 1 and 2 of docs/the project documentation: from SSM/I brightness
temperature and an external skin temperature, derive a per-channel surface
emissivity on clear-sky pixels, after a first-order clear-sky atmospheric
correction. The two external inputs are PLUG-INS, so the harness runs today on
F-13 microwave alone and is ready the moment Ken Knapp's GridSat-B1 product
arrives:

    skin temperature  Ts          -> Ken's GridSat-B1 cloud-cleared T_s
    clear-sky mask    clear        -> Ken's ESDS machine-learning cloud mask
    column water vapour  tcwv      -> ERA5 (optional; refines the correction)

Physics (single isothermal layer, no scattering, Prigent and Rossow lineage).
Top-of-atmosphere brightness temperature for a surface (emissivity e, skin
temperature Ts) under a layer of transmissivity tau and effective temperature
Ta:

    Tb = tau*e*Ts + tau*(1-e)*Td + Tu
    Tu = Ta*(1-tau)                      upwelling atmospheric emission
    Td = Ta*(1-tau)                      downwelling, reflected by the surface
                                         (the tau*T_cmb term is dropped here)

Solving for the surface emissivity, with A = Ta*(1-tau):

    e = (Tb - A - tau*A) / (tau*(Ts - A))

The nominal transmissivities and absorption coefficients below are first-order
placeholders chosen so the window channels stay near-transparent and 22 GHz (on
the water-vapour line) is the most opaque. They are good enough to show the dry
versus wet emissivity contrast (Step 2); a MonoRTM-class forward model replaces
them only if the first cut is promising (Step 6).
"""

import numpy as np

from .channels import N_CHANNELS, CHANNEL_NAMES

# SSM/I Earth-incidence angle. The slant path lengthens the optical depth by
# this air-mass factor relative to nadir.
INCIDENCE_DEG = 53.1
AIRMASS = 1.0 / np.cos(np.deg2rad(INCIDENCE_DEG))

# Nominal clear-sky one-way nadir optical depth per Basist channel, split into a
# dry/oxygen baseline and a water-vapour term (per mm of column water vapour).
# Order: 19V 19H 22V 37V 37H 85V 85H.
TAU_DRY = np.array([0.030, 0.030, 0.040, 0.050, 0.050, 0.100, 0.100])
KWV_PER_MM = np.array([0.0007, 0.0007, 0.0040, 0.0015, 0.0015, 0.0060, 0.0060])

# Nominal transmissivity used when no column water vapour is supplied (a dry,
# morning-overpass first cut: ~25 mm pwv folded into the baseline).
NOMINAL_TCWV_MM = 25.0


def transmissivity(tcwv_mm=None):
    """Per-channel clear-sky transmissivity (length-7 vector).

    tcwv_mm may be a scalar or an array of total column water vapour in mm. With
    None, NOMINAL_TCWV_MM is used. Returns transmissivity broadcast over the
    input shape with a trailing length-7 channel axis.
    """
    if tcwv_mm is None:
        tcwv_mm = NOMINAL_TCWV_MM
    tcwv = np.asarray(tcwv_mm, dtype=np.float64)
    # one-way nadir optical depth, then slant path
    tau_opt = TAU_DRY + tcwv[..., np.newaxis] * KWV_PER_MM
    return np.exp(-AIRMASS * tau_opt)


def effective_atm_temperature(ts, lapse=10.0):
    """First-order single-layer effective atmospheric temperature (K).

    A fixed offset below the skin temperature. The morning overpass samples the
    diurnal minimum where this approximation is least bad.
    """
    return np.asarray(ts, dtype=np.float64) - lapse


def emissivity_single_layer(tb, ts, tau, t_atm):
    """Invert the isothermal single-layer clear-sky model for emissivity.

    All arguments broadcast. Returns e = (Tb - A - tau*A) / (tau*(Ts - A)) with
    A = t_atm*(1 - tau). NaN where the denominator vanishes (Ts == A).
    """
    tb = np.asarray(tb, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    a = np.asarray(t_atm, dtype=np.float64) * (1.0 - tau)
    num = tb - a - tau * a
    den = tau * (ts - a)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = num / den
    e[~np.isfinite(e)] = np.nan
    return e


def apparent_emissivity(tb, ts):
    """Zeroth-order apparent emissivity Tb/Ts (no atmospheric correction).

    Cheap sanity check for the dry-versus-wet contrast before the single-layer
    correction is applied.
    """
    tb = np.asarray(tb, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)[..., np.newaxis] if np.ndim(ts) == np.ndim(tb) - 1 else np.asarray(ts, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return tb / ts


def derive_emissivity(tb, ts, clear=None, tcwv_mm=None, t_atm=None, lapse=10.0):
    """Per-channel surface emissivity over clear-sky pixels (Steps 1 and 2).

    Parameters
    ----------
    tb : (..., 7) array, brightness temperature in Kelvin (Basist channel order).
    ts : (...) array, surface skin temperature in Kelvin. PLUG-IN: Ken's
         GridSat-B1 cloud-cleared T_s. Must match tb's leading shape.
    clear : (...) bool array or None. True where the pixel is clear sky. PLUG-IN:
         Ken's ESDS machine-learning cloud mask. None treats every valid pixel as
         clear (development only).
    tcwv_mm : (...) array, scalar, or None. Total column water vapour in mm.
         PLUG-IN: ERA5 (task b). None uses the nominal dry-atmosphere default.
    t_atm : (...) array or None. Effective single-layer atmospheric temperature in
         Kelvin (for example ERA5 2 m temperature). None falls back to ts - lapse.
    lapse : effective skin-to-atmosphere temperature offset (K), used when t_atm
         is None.

    Returns
    -------
    dict with:
      emissivity : (..., 7) per-channel emissivity, NaN off clear sky / invalid.
      apparent   : (..., 7) zeroth-order Tb/Ts for comparison.
      tau        : (..., 7) transmissivity actually used.
      clear      : (...) bool mask actually applied.
      n_clear    : int count of pixels solved.
    """
    tb = np.asarray(tb, dtype=np.float64)
    if tb.shape[-1] != N_CHANNELS:
        raise ValueError(f"tb last axis must be {N_CHANNELS}, got {tb.shape[-1]}")
    ts = np.asarray(ts, dtype=np.float64)
    if ts.shape != tb.shape[:-1]:
        raise ValueError(f"ts shape {ts.shape} must match tb leading shape {tb.shape[:-1]}")

    if clear is None:
        clear = np.isfinite(tb).all(axis=-1) & np.isfinite(ts)
    else:
        clear = np.asarray(clear, dtype=bool) & np.isfinite(tb).all(axis=-1) & np.isfinite(ts)

    tau = transmissivity(tcwv_mm)
    tau = np.broadcast_to(tau, tb.shape).copy()
    if t_atm is None:
        t_atm_field = effective_atm_temperature(ts, lapse=lapse)
    else:
        t_atm_field = np.asarray(t_atm, dtype=np.float64)
        if t_atm_field.shape != ts.shape:
            raise ValueError(f"t_atm shape {t_atm_field.shape} must match ts {ts.shape}")
    t_atm_b = t_atm_field[..., np.newaxis]

    e = emissivity_single_layer(tb, ts[..., np.newaxis], tau, t_atm_b)
    app = np.asarray(tb / ts[..., np.newaxis], dtype=np.float64)

    # mask everything off clear sky
    off = ~clear[..., np.newaxis]
    e[np.broadcast_to(off, e.shape)] = np.nan
    app[np.broadcast_to(off, app.shape)] = np.nan

    return {
        "emissivity": e,
        "apparent": app,
        "tau": tau,
        "clear": clear,
        "n_clear": int(clear.sum()),
        "channels": list(CHANNEL_NAMES),
    }


def contrast_summary(result):
    """Compact per-channel emissivity summary (median and IQR over clear sky).

    Returns a list of dicts, one per channel, for a quick dry-versus-wet read.
    """
    e = result["emissivity"]
    out = []
    for i, name in enumerate(result["channels"]):
        v = e[..., i]
        v = v[np.isfinite(v)]
        if v.size:
            out.append({
                "channel": name,
                "n": int(v.size),
                "p10": float(np.percentile(v, 10)),
                "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)),
            })
        else:
            out.append({"channel": name, "n": 0,
                        "p10": np.nan, "median": np.nan, "p90": np.nan})
    return out
