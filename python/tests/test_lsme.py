"""Round-trip and sanity tests for the LSME first-order estimator."""

import numpy as np
import pytest

from swi import lsme
from swi.channels import N_CHANNELS


def _forward(e, ts, tau, t_atm):
    """Forward single-layer model: emissivity -> brightness temperature."""
    a = t_atm * (1.0 - tau)
    return tau * e * ts + tau * (1.0 - e) * a + a


def test_inversion_recovers_known_emissivity():
    """Forward-model a known emissivity field, then invert and recover it."""
    rng = np.random.default_rng(0)
    shape = (50, 40)
    e_true = rng.uniform(0.4, 0.98, size=shape + (N_CHANNELS,))
    ts = rng.uniform(270.0, 305.0, size=shape)
    tau = lsme.transmissivity(20.0)                      # length-7
    tau = np.broadcast_to(tau, shape + (N_CHANNELS,))
    t_atm = lsme.effective_atm_temperature(ts)[..., np.newaxis]

    tb = _forward(e_true, ts[..., np.newaxis], tau, t_atm)
    e_rec = lsme.emissivity_single_layer(tb, ts[..., np.newaxis], tau, t_atm)

    assert np.allclose(e_rec, e_true, atol=1e-9)


def test_derive_masks_off_cloudy_pixels():
    """Pixels flagged not-clear come back NaN; clear ones are finite."""
    shape = (8, 8)
    tb = np.full(shape + (N_CHANNELS,), 250.0)
    ts = np.full(shape, 290.0)
    clear = np.zeros(shape, dtype=bool)
    clear[::2, ::2] = True

    r = lsme.derive_emissivity(tb, ts, clear=clear)
    assert r["n_clear"] == int(clear.sum())
    assert np.isfinite(r["emissivity"][clear]).all()
    assert np.isnan(r["emissivity"][~clear]).all()


def test_wet_surface_has_lower_emissivity_than_dry():
    """Lower Tb at fixed Ts must invert to lower emissivity (the core signal)."""
    ts = np.array([290.0, 290.0])
    tb = np.empty((2, N_CHANNELS))
    tb[0, :] = 275.0      # dry-like: warm, high emissivity
    tb[1, :] = 200.0      # wet-like: cold, low emissivity
    r = lsme.derive_emissivity(tb, ts)
    e = r["emissivity"]
    assert np.all(e[0] > e[1])


def test_supplied_t_atm_overrides_the_lapse_default():
    """An explicit t_atm field is used instead of ts - lapse, and changes e."""
    ts = np.full((6, 6), 295.0)
    tb = np.full((6, 6, N_CHANNELS), 260.0)
    e_lapse = lsme.derive_emissivity(tb, ts)["emissivity"]
    e_tatm = lsme.derive_emissivity(tb, ts, t_atm=np.full((6, 6), 270.0))["emissivity"]
    assert np.isfinite(e_lapse).all() and np.isfinite(e_tatm).all()
    assert not np.allclose(e_lapse, e_tatm)


def test_t_atm_shape_must_match_ts():
    ts = np.full((6, 6), 295.0)
    tb = np.full((6, 6, N_CHANNELS), 260.0)
    with pytest.raises(ValueError):
        lsme.derive_emissivity(tb, ts, t_atm=np.full((5, 5), 270.0))


def test_scattering_index_and_keep_mask():
    """SI37 = 37V - 85V, and the keep-mask drops only high-depression pixels."""
    from swi.channels import CH37V, CH85V
    tb = np.full((3, 3, N_CHANNELS), 260.0)
    # One scattering pixel: 85V driven well below 37V (a 20 K depression).
    tb[1, 1, CH37V] = 265.0
    tb[1, 1, CH85V] = 245.0
    si = lsme.scattering_index_si37(tb)
    assert si[1, 1] == 20.0
    assert si[0, 0] == 0.0
    keep = lsme.scattering_keep_mask(tb, threshold_k=8.0)
    assert keep[0, 0] and not keep[1, 1]


def test_scatter_screen_kwarg_drops_scattering_pixels_only():
    """scatter_screen_k removes the scattering pixel; default off is unchanged."""
    from swi.channels import CH37V, CH85V
    ts = np.full((3, 3), 290.0)
    tb = np.full((3, 3, N_CHANNELS), 270.0)
    tb[1, 1, CH85V] = 245.0          # 37V - 85V = 25 K depression
    tb[1, 1, CH37V] = 270.0
    off = lsme.derive_emissivity(tb, ts)
    on = lsme.derive_emissivity(tb, ts, scatter_screen_k=8.0)
    # Default off keeps every valid pixel.
    assert np.isfinite(off["emissivity"][1, 1]).all()
    # Screen on drops the scattering pixel only.
    assert np.isnan(on["emissivity"][1, 1]).all()
    assert np.isfinite(on["emissivity"][0, 0]).all()
    assert on["n_clear"] == off["n_clear"] - 1


def test_scatter_screen_none_is_bit_identical():
    """scatter_screen_k=None reproduces the unscreened result exactly."""
    rng = np.random.default_rng(3)
    ts = rng.uniform(280, 300, size=(8, 8))
    tb = rng.uniform(230, 285, size=(8, 8, N_CHANNELS))
    a = lsme.derive_emissivity(tb, ts)["emissivity"]
    b = lsme.derive_emissivity(tb, ts, scatter_screen_k=None)["emissivity"]
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_tune_si37_threshold_picks_a_data_based_keep_fraction():
    """The tuned threshold should keep about target_keep_fraction of pixels."""
    rng = np.random.default_rng(0)
    # Gaussian-ish si37 around 0 K, with a heavy positive tail for scattering.
    n = 5000
    tb = np.zeros((n, 7))
    tb[:, 3] = 220.0 + rng.normal(0, 2, n)                   # 37V
    tail_size = n // 50
    si85_offset = np.zeros(n)
    si85_offset[:tail_size] = rng.uniform(10, 30, tail_size)  # ice-scattering tail
    tb[:, 5] = tb[:, 3] - rng.normal(0, 2, n) - si85_offset   # 85V depressed by tail
    out = lsme.tune_si37_threshold(tb, target_keep_fraction=0.95)
    # Roughly 95 percent kept by construction. n_tune is the finite pixel count.
    assert 0.93 <= out["keep_fraction"] <= 0.97
    assert out["n_tune"] == n
    # The threshold should be in the scattering-tail range (well above 0).
    assert 2.0 < out["threshold_k"] < 25.0


def test_tune_si37_threshold_evaluates_candidates():
    """The candidates list reports each candidate's keep-fraction in the tune data."""
    rng = np.random.default_rng(1)
    n = 3000
    tb = np.zeros((n, 7))
    tb[:, 3] = 250.0 + rng.normal(0, 1, n)
    tb[:, 5] = tb[:, 3] - rng.normal(2, 3, n)
    out = lsme.tune_si37_threshold(tb, candidates=[0, 5, 10, 20],
                                   target_keep_fraction=0.99)
    assert "candidates" in out and len(out["candidates"]) == 4
    # Larger thresholds keep more pixels (monotonic).
    fracs = [k_frac for _, k_frac in out["candidates"]]
    assert all(fracs[i] <= fracs[i + 1] for i in range(len(fracs) - 1))
