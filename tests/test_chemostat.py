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


def test_herbivorous_forams_eat_only_phytoplankton(model):
    """For bn, bs, sn (herbivorous forams): Z and F prey rows are zero in their predator column."""
    for jpred in np.flatnonzero(model.PFT_F_bool):
        if not model.is_herbivore[jpred] or model.is_carnivore[jpred]:
            continue
        assert np.all(model.f[model.PFT_Z_bool, jpred] == 0.0)
        assert np.all(model.f[model.PFT_F_bool, jpred] == 0.0)


def test_carnivorous_ss_eats_only_zooplankton(model):
    """ss is the only carnivorous foram in FORAMECOGEM: phyto and foram prey rows are zero."""
    ss_idx = [j for j in np.flatnonzero(model.PFT_F_bool) if model.foram_type[j] == 'ss']
    assert len(ss_idx) > 0, 'expected at least one ss foram in default config'
    for jpred in ss_idx:
        assert np.all(model.f[model.PFT_P_bool, jpred] == 0.0)
        assert np.all(model.f[model.PFT_F_bool, jpred] == 0.0)
        # Diameter-based ECOGEM kernel; spine_scale^1 (not ^3) for geometric consistency
        zoo_idx = np.flatnonzero(model.PFT_Z_bool)
        sig = model.sigma_f * model.foram_spine_scale
        thet = model.theta_opt * model.foram_spine_scale
        expected = model._cal_platability(model.diameter[jpred], model.diameter[zoo_idx], sig, thet)
        assert np.allclose(model.f[zoo_idx, jpred], expected)


def test_foram_predator_kernel_uses_sigma_f(model):
    """Non-spinose foram predator columns use sigma_f (not sigma_z) for the diameter-based prey kernel."""
    P_idx = np.flatnonzero(model.PFT_P_bool)
    for jpred in np.flatnonzero(model.PFT_F_bool):
        if model.is_spinose[jpred] or not model.is_herbivore[jpred]:
            continue
        expected = model._cal_platability(
            model.diameter[jpred], model.diameter[P_idx], model.sigma_f, model.theta_opt,
        )
        assert np.allclose(model.f[P_idx, jpred], expected)


def test_grazing_matrix_zoo_on_foram_scaled_by_cal_pg(model):
    """Zoo predator × foram prey entries equal cal_pg * diameter-based palatability."""
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
    """Gmax[F] = allom * per-morphotype growthcost_factor (stacked with foram_hetero_cost for sn/ss)."""
    P, Z, F = model.PFT_P_bool, model.PFT_Z_bool, model.PFT_F_bool
    allom = model.Gmax_a * model.Vol ** model.Gmax_b
    assert np.allclose(model.Gmax[P], 0.0)
    assert np.allclose(model.Gmax[Z], allom[Z])
    expected_F = allom[F] * model.foram_growthc[F]
    sym_F = model.is_symbiont[F]
    expected_F = np.where(sym_F, expected_F * model.foram_hetero_cost, expected_F)
    assert np.allclose(model.Gmax[F], expected_F)


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


# ---------- foram morphotype tests (symbiotic / spinose port from ECOGEM) ----------

def _rebuild_with_types(model, types):
    """Patch in a custom type_foram and rebuild the derived state."""
    model.type_foram = list(types)
    model._init_foram_types()
    model._init_array()
    model._assign_plank_par()
    return model


def test_foram_morphotype_arrays_match_config(model):
    """Each foram pft inherits ns / growthcost / calcite_respir / feeding mode from its morphotype."""
    expected = {
        'bn': dict(ns=1.0, growthc=1.3, calc=0.02,  herb=True,  carn=False),
        'bs': dict(ns=5.0, growthc=1.0, calc=0.045, herb=True,  carn=False),
        'sn': dict(ns=1.0, growthc=1.3, calc=0.02,  herb=True,  carn=False),
        'ss': dict(ns=5.0, growthc=1.0, calc=0.045, herb=False, carn=True),
    }
    for j in np.flatnonzero(model.PFT_F_bool):
        e = expected[model.foram_type[j]]
        assert model.foram_ns[j]        == pytest.approx(e['ns'])
        assert model.foram_growthc[j]   == pytest.approx(e['growthc'])
        assert model.foram_calc_resp[j] == pytest.approx(e['calc'])
        assert bool(model.is_herbivore[j]) is e['herb']
        assert bool(model.is_carnivore[j]) is e['carn']


def test_foram_prey_switching_raises_kernel_times_biomass_to_ns(model):
    """Foram switching weight is (f_j*B_j)^ns / sum_k (f_k*B_k)^ns, as in ECOGEM's food_ns."""
    j = np.flatnonzero(model.PFT_F_bool)[0]
    model.foram_ns[j] = 5.0
    f_j = model.f[:, j]
    # two phyto prey the foram can eat, with different kernel values
    prey = [p for p in np.flatnonzero(model.PFT_P_bool) if f_j[p] > 0.05][:2]
    assert len(prey) == 2 and f_j[prey[0]] != pytest.approx(f_j[prey[1]])
    B = np.zeros(model.n_pft)
    B[prey] = 0.3
    B[j] = 0.01
    gamma_T = np.exp(model.R * (model.T - model.T_ref))
    dBdt = np.zeros(model.n_pft)
    model._foram_graze_one(j, B, gamma_T, dBdt)

    F = np.sum(f_j * B)
    total = model.Gmax[j] * gamma_T * F / (F + model.knprey) * (1.0 - np.exp(model.lamdaprey * F))
    w = (f_j[prey] * B[prey]) ** 5.0
    graze = total * w / w.sum()
    assert dBdt[prey] == pytest.approx(-graze * B[j])
    assert dBdt[j] == pytest.approx(graze.sum() * B[j] * model.lamda)


def test_foram_calcite_respiration_added(model):
    """resp[F] = resp_p + per-morphotype calcite_respir; spinose pay more."""
    for j in np.flatnonzero(model.PFT_F_bool):
        assert model.resp[j] == pytest.approx(model.resp_p + model.foram_calc_resp[j])


def test_foram_symbiont_uptake_is_active_for_sn_ss(model):
    """sn / ss get nonzero mumax derived from a tiny auto-volume, scaled by foram_auto_cost."""
    sym_F = np.flatnonzero(model.is_symbiont)
    assert len(sym_F) > 0, 'expected sn/ss in default config'
    assert np.all(model.mumax[sym_F] > 0.0)
    assert np.all(model.kN[sym_F] > 0.0)
    # mumax for symbiont forams should be much smaller than for similarly sized phytoplankton
    P = np.flatnonzero(model.PFT_P_bool)
    assert model.mumax[sym_F].max() < model.mumax[P].max()
    # bn / bs (non-symbiont) forams take up no N
    nonsym_F = np.array([j for j in np.flatnonzero(model.PFT_F_bool) if not model.is_symbiont[j]])
    assert np.allclose(model.mumax[nonsym_F], 0.0)


def test_foram_spine_widens_prey_selection_kernel(model):
    """Spinose forams (bs, ss) at a given size graze across a much wider prey-size range."""
    nF = model.n_foram
    base = _rebuild_with_types(model, ['bn'] * nF)
    bn_sum_f = base.f[:, base.PFT_F_bool].sum(axis=0).copy()
    _rebuild_with_types(model, ['bs'] * nF)
    bs_sum_f = model.f[:, model.PFT_F_bool].sum(axis=0)
    assert np.all(bs_sum_f > bn_sum_f)


def test_invalid_foram_type_raises(config_path):
    """Unknown morphotype tags are rejected up front."""
    from chemostat import Chemostat
    m = Chemostat(20.0, 2.5)
    m.load_ecoconfig(str(config_path))
    m.type_foram = ['bn'] * (m.n_foram - 1) + ['xx']
    with pytest.raises(ValueError, match='unknown foram type'):
        m._init_foram_types()


def test_diff_eqn_runs_without_nan(model):
    """A representative multi-PFT state should produce finite, non-NaN dydt under FORAMECOGEM
    config. Replaces the legacy pinned-numeric regression test (whose indices and numbers were
    tied to the 14-foram size spectrum, no longer the default)."""
    nP, nZ, nF = model.n_phytoplankton, model.n_zooplankton, model.n_foram
    B = np.zeros(model.n_pft)
    B[5] = 0.30
    B[15] = 0.05
    B[nP + 5] = 0.20
    B[nP + 15] = 0.02
    for k in range(nF):
        B[nP + nZ + k] = 0.01
    y = np.concatenate(([12.0], B))
    dydt = model.diff_eqn(y, 0.0)
    assert np.all(np.isfinite(dydt))
    # ss must receive prey from the zoo column it actually feeds on
    ss_idx = [j for j in np.flatnonzero(model.PFT_F_bool) if model.foram_type[j] == 'ss']
    if ss_idx:
        # ss biomass change is nontrivial when zoo prey are present
        assert dydt[1 + ss_idx[0]] != 0.0
