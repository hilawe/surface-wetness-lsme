"""Anomaly form of the LSME emissivity and the two-estimator wetness product.

The absolute monthly emissivity, and the relative wetness index built from it, is
dominated by the static surface-type pattern: bare deserts emit high, dense
forests lower, open water lowest, and that pattern is essentially the same every
month. It inflates the month-to-month correlation against a climatology (a desert
is a desert in January and in July) without saying anything about change. A
monitoring climate data record is about the time-varying part, not the fixed map.

The anomaly form removes each cell's temporal mean and keeps the departure. For a
single year's twelve monthly fields that departure is the seasonal cycle (each
month relative to the annual mean); once several years are in hand the same
operation gives the interannual anomaly. Either way it turns the relative index
into a time-varying wetness signal and supports a harder, more honest validation:
does the derived emissivity track the *variation* of the TELSEM climatology
through the year, not merely reproduce its fixed spatial pattern.

The functions operate on a stack with time as the leading axis, shape
(T, ...) where the trailing axes are spatial (and optionally per-channel). They
are shape-agnostic on the trailing axes, so the same code serves a single scalar
wetness field (T, nlat, nlon) and a per-channel emissivity stack
(T, nlat, nlon, nchannel). The spatial-skill helpers expect 2-D fields, so the
caller loops channels when validating per channel.
"""

import numpy as np

from . import validate


def temporal_mean(stack, min_valid=2):
    """Per-cell mean over the valid (finite) entries of a (T, ...) stack.

    Cells with fewer than min_valid finite time steps are NaN, because a mean
    (and therefore an anomaly) is not meaningfully defined from a single value.
    """
    a = np.asarray(stack, np.float64)
    valid = np.isfinite(a)
    n = valid.sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(valid, a, np.nan), axis=0)
    return np.where(n >= min_valid, mean, np.nan)


def temporal_anomaly(stack, min_valid=2):
    """Departure of each time step from the per-cell temporal mean.

    stack is (T, ...). Returns an anomaly stack of the same shape. An entry is
    NaN where its own value is missing, and a whole cell is NaN where it has
    fewer than min_valid finite time steps. The trailing-axis mean broadcasts
    against the leading time axis, so this works for any trailing shape.
    """
    a = np.asarray(stack, np.float64)
    return a - temporal_mean(a, min_valid=min_valid)


def pattern_skill(field_a, field_b, mask=None):
    """Spatial skill of two 2-D fields over finite (and optionally masked) cells.

    Returns the validate.skill_scores dict (n, pearson_r, spearman_r, bias,
    rmse) computed over the cells where both fields are finite and the mask, if
    given, is true.
    """
    a = np.asarray(field_a, np.float64)
    b = np.asarray(field_b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    return validate.skill_scores(a[m], b[m])


def anomaly_pattern_skill(anom_a, anom_b, mask=None):
    """Spatial pattern skill of two anomaly stacks, per time step and pooled.

    anom_a and anom_b are (T, nlat, nlon) anomaly stacks (for one channel or one
    scalar field). Returns (per_time, aggregate):

      per_time   a list of T dicts, each the pattern_skill for that time step
                 with an added "t" index.
      aggregate  the skill_scores over every (time, cell) anomaly pair pooled
                 together. This is the headline number: with the static spatial
                 mean removed, does the derived field reproduce the reference's
                 month-to-month variation across the whole record.
    """
    a = np.asarray(anom_a, np.float64)
    b = np.asarray(anom_b, np.float64)
    if a.shape != b.shape:
        raise ValueError("anomaly stacks must have the same shape")

    per_time = []
    for t in range(a.shape[0]):
        s = pattern_skill(a[t], b[t], mask=mask)
        s["t"] = t
        per_time.append(s)

    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)[None, :, :]
    aggregate = validate.skill_scores(a[m], b[m])
    return per_time, aggregate


def block_mean(a, n, axis):
    """NaN-aware block average along one axis by an integer factor n.

    Trims the axis to a multiple of n, groups it into blocks of n, and averages
    each block while ignoring NaN. Apply once per spatial axis to coarsen a 0.25
    degree field to 0.5 degree (n=2) or 1.0 degree (n=4). Coarsening averages
    across the orbital swath-sampling stripes that show up in a per-cell map at
    the native resolution. n=1 returns the input unchanged.
    """
    a = np.asarray(a, np.float64)
    if n <= 1:
        return a
    keep = (a.shape[axis] // n) * n
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(0, keep)
    a = a[tuple(sl)]
    shape = a.shape[:axis] + (keep // n, n) + a.shape[axis + 1:]
    with np.errstate(invalid="ignore"):
        return np.nanmean(a.reshape(shape), axis=axis + 1)
