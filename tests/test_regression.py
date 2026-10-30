from pathlib import Path

import numpy as np
import pytest

from chemostat import Chemostat


@pytest.mark.parametrize('config, temperature', [
    ('model_config.toml', 20.0),
    ('model_config_tuned.toml', 30.0),
])
def test_simulation_matches_before_modularization(config_path, config, temperature):
    """Reference: original model, 120 days, six samples, source N=2.5.

    The derivative uses random seed 42 to exercise a mixed community alongside
    the simulation from the configured initial state.
    """
    model = Chemostat(temperature, 2.5)
    model.load_ecoconfig(config_path.parent / config)
    model.nday = 120
    model.save_freq = 6
    model._init_array()
    model._assign_plank_par()
    state = np.concatenate((
        [12.0], np.random.default_rng(42).uniform(0, 0.3, model.n_pft),
    ))
    derivative = model.diff_eqn(state, 0.0)
    model.run()

    with np.load(Path(__file__).parent / 'data' / 'pre_modularization.npz') as reference:
        prefix = Path(config).stem
        np.testing.assert_allclose(
            derivative, reference[f'{prefix}_derivative'], rtol=1e-12, atol=1e-14,
        )
        for name in ('output_t', 'output_N', 'output_B', 'output_size', 'output_pft_richness'):
            np.testing.assert_allclose(
                getattr(model, name), reference[f'{prefix}_{name}'], rtol=1e-8, atol=1e-12,
            )
