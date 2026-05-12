import numpy as np
import pytest

from chemostat import Chemostat


# ---------- trait-level tests ----------

def test_pcmax_a_step_function(model):
    """pcmax_a follows the 4-step size rule for every configured phyto diameter."""
    expected = np.array([model._cal_pcmax_a(d) for d in model.esd_phytoplankton])
    assert np.array_equal(model.pcmax_a[: model.n_phytoplankton], expected)
    # entries beyond the phyto block are never assigned — default zero
    assert np.all(model.pcmax_a[model.n_phytoplankton :] == 0.0)


def test_palatability_matches_formula():
    """_cal_platability is a Gaussian on ln(size ratio / theta_opt)."""
    m = Chemostat(0, 0)
    pred, prey, sigma, theta = 9.9264, 158.8224, 0.5, 10.079368391360413
    expected = np.exp(-(np.log((pred / prey) / theta) ** 2) / (2 * sigma ** 2))
    assert m._cal_platability(pred, prey, sigma, theta) == pytest.approx(expected)


# ---------- grazing matrix structural tests ----------

def test_grazing_matrix_phyto_predator_columns_are_zero(model):
    """Phytoplankton never graze — their predator columns stay zero."""
    assert np.all(model.f[:, model.PFT_P_bool] == 0.0)


def test_grazing_matrix_foram_predators_only_eat_phytoplankton(model):
    """Foram predator columns are zero on Z/F prey rows and equal the bare palatability on P prey rows."""
    f_F_pred = model.f[:, model.PFT_F_bool]
    non_phyto = ~model.PFT_P_bool
    assert np.all(f_F_pred[non_phyto] == 0.0)

    phyto_idx = np.flatnonzero(model.PFT_P_bool)
    for jpred in np.flatnonzero(model.PFT_F_bool):
        expected = model._cal_platability(
            model.diameter[jpred], model.diameter[phyto_idx],
            model.sigma_z, model.theta_opt,
        )
        assert np.allclose(model.f[phyto_idx, jpred], expected)


def test_grazing_matrix_zoo_on_foram_scaled_by_cal_pg(model):
    """Zoo predator × foram prey entries equal cal_pg * palatability."""
    foram_idx = np.flatnonzero(model.PFT_F_bool)
    for jpred in np.flatnonzero(model.PFT_Z_bool):
        expected = model.cal_pg * model._cal_platability(
            model.diameter[jpred], model.diameter[foram_idx],
            model.sigma_z, model.theta_opt,
        )
        assert np.allclose(model.f[foram_idx, jpred], expected)


def test_grazing_matrix_diagonal_is_zero_without_cannibalism(model):
    assert model.cannibalism is False
    assert np.all(np.diag(model.f) == 0.0)


# ---------- Gmax assignment tests ----------

def test_gmax_zero_for_phyto_scaled_for_foram(model):
    P, Z, F = model.PFT_P_bool, model.PFT_Z_bool, model.PFT_F_bool
    allom = model.Gmax_a * model.Vol ** model.Gmax_b
    assert np.allclose(model.Gmax[P], 0.0)
    assert np.allclose(model.Gmax[Z], allom[Z])
    assert np.allclose(model.Gmax[F], allom[F] * model.cal_cost)


# ---------- diff_eqn invariant tests ----------

def test_diff_eqn_zero_biomass(model):
    """With B=0, dN/dt = K*(source_N - N) and all dB/dt = 0."""
    y = np.zeros(1 + model.n_pft)
    y[0] = 2.0
    dydt = model.diff_eqn(y, 0.0)
    assert dydt[0] == pytest.approx(model.K * (model.source_N - y[0]))
    assert np.allclose(dydt[1:], 0.0)


def test_diff_eqn_single_phyto_only_grows_via_uptake(model):
    """A lone phyto cell sees uptake + dilution + mortality + respiration, no grazing."""
    i = 5
    assert model.PFT_P_bool[i]
    N0, B0 = 1.0, 0.1
    y = np.zeros(1 + model.n_pft); y[0] = N0; y[1 + i] = B0
    dydt = model.diff_eqn(y, 0.0)

    gamma_T = np.exp(model.R * (model.T - model.T_ref))
    uptake = model.gamma_l * gamma_T * model.mumax[i] * N0 / (model.kN[i] + N0) * B0
    losses = B0 * model.m[i] + model.K * B0 + B0 * model.resp[i] * gamma_T
    assert dydt[1 + i] == pytest.approx(uptake - losses)

    others = np.ones(model.n_pft, dtype=bool); others[i] = False
    assert np.allclose(dydt[1:][others], 0.0)


def test_diff_eqn_regression(model):
    """Pin dydt on a fixed multi-PFT state. Numbers are tied to model_config.toml;
    if the config or the math changes, regenerate by re-running diff_eqn on this y."""
    nP, nZ = model.n_phytoplankton, model.n_zooplankton
    B = np.zeros(model.n_pft)
    B[5] = 0.30
    B[15] = 0.05
    B[nP + 5] = 0.20
    B[nP + 15] = 0.02
    B[nP + nZ + 2] = 0.01
    B[nP + nZ + 8] = 0.005
    y = np.concatenate(([12.0], B))
    dydt = model.diff_eqn(y, 0.0)

    P, Z, F = model.PFT_P_bool, model.PFT_Z_bool, model.PFT_F_bool
    # dN clamped to 0: at N=12, K*(source_N - N) is negative and uptake also drains N
    assert dydt[0] == pytest.approx(0.0)
    assert np.sum(dydt[1:][P]) == pytest.approx(-0.2051259862823555, rel=1e-9)
    assert np.sum(dydt[1:][Z]) == pytest.approx(0.14758621193359717, rel=1e-9)
    assert np.sum(dydt[1:][F]) == pytest.approx(0.004549162256240617, rel=1e-9)
    assert float(np.max(np.abs(dydt[1:]))) == pytest.approx(0.20680399732708085, rel=1e-9)
