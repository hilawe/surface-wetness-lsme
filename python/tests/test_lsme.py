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
