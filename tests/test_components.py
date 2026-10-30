import numpy as np
import pytest

from chemostat import Chemostat
from chemostat.components import foram_grazing, nutrient_uptake, zooplankton_grazing


def test_uptake_uses_only_enabled_groups():
    biomass = np.array([1, 2, 3])
    uptake = nutrient_uptake(
        2.0, biomass, np.array([1.0, 1.0, 2.0]),
        light=np.array([0.1, 0.1, 1.0]), mumax=np.array([2.0, 0.0, 0.5]),
        kN=np.array([2.0, 0.0, 2.0]), can_uptake=np.array([True, False, True]),
    )
    np.testing.assert_allclose(uptake, [0.1, 0.0, 1.5])
    np.testing.assert_array_equal(biomass, [1, 2, 3])


@pytest.mark.parametrize('graze', [zooplankton_grazing, foram_grazing])
def test_grazing_transfers_assimilated_prey_to_predator(graze):
    biomass = np.array([2.0, 3.0, 5.0, 0.4])
    change = np.zeros(4)
    options = (
        {'prey_groups': (
            np.array([True, False, False, False]),
            np.array([False, True, False, True]),
            np.array([False, False, True, False]),
        )}
        if graze is zooplankton_grazing else {'switching_exponent': 5.0}
    )
    graze(
        3, biomass, 1.5, change, preference=np.array([1.0, 0.5, 0.25, 0.0]),
        Gmax=1.2, refuge_strength=-1.0, half_saturation=0.1,
        assimilation=0.7, **options,
    )
    assert np.all(change[:3] < 0)
    assert change[3] == pytest.approx(-change[:3].sum() * 0.7)
    np.testing.assert_array_equal(biomass, [2.0, 3.0, 5.0, 0.4])


def test_replacing_uptake_changes_biomass_and_nutrient_together(model):
    state = np.full(model.n_pft + 1, 0.1)
    state[0] = 2.0
    original_change = model.diff_eqn(state, 0.0)
    gamma_T = np.exp(model.R_pft * (model.T - model.T_ref))
    uptake = model._phyto_uptake(state[0], state[1:], gamma_T)
    original_uptake = model.processes['uptake']

    def half_uptake(*args, **kwargs):
        return 0.5 * original_uptake(*args, **kwargs)

    model.processes['uptake'] = half_uptake
    change = model.diff_eqn(state, 0.0) - original_change
    assert change[0] == pytest.approx(0.5 * uptake.sum())
    np.testing.assert_allclose(change[1:], -0.5 * uptake, atol=1e-15)
    assert Chemostat(20.0, 2.5).processes['uptake'] is original_uptake
