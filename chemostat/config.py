"""Load TOML parameters and initialise the plankton community."""

import tomllib

import numpy as np


def load_ecoconfig(model, file):
    "read config from file, set related parameters, and initialise arrays"
    with open(file, "rb") as f:
        data = tomllib.load(f)

    # set hyper config
    for name, value in data.get('parameters', {}).items():
        setattr(model, name, value)

    # set plankton structure
    for name, value in data.get('settings', {}).items():
        setattr(model, name, value)

    # per-morphotype FORAMECOGEM calibration (.zoo file row per type)
    model.foram_morphotype = data.get('foram_morphotype', {})

    # solver options (optional section); step sizes are left to SciPy, since a
    # min_step floor makes LSODA fail when tighter tolerances need smaller steps
    model.solver_opts = {
        'method': 'LSODA',
        'rtol': 1e-3,
        'atol': 1e-9,
        'max_step': 28.0,
        **data.get('solver', {}),
    }

    model.n_phytoplankton = len(model.esd_phytoplankton)
    model.n_zooplankton = len(model.esd_zooplankton)
    model.n_foram = len(model.esd_foram)

    model.diameter = np.concatenate([model.esd_phytoplankton,model.esd_zooplankton, model.esd_foram])


    model.n_pft = model.n_phytoplankton + model.n_zooplankton + model.n_foram

    model.PFT_P=np.zeros(model.n_pft)
    model.PFT_Z=np.zeros(model.n_pft)
    model.PFT_F=np.zeros(model.n_pft)
    model.PFT_P_bool = np.zeros(model.n_pft)
    model.PFT_Z_bool = np.zeros(model.n_pft)
    model.PFT_F_bool = np.zeros(model.n_pft)


    ## loop over plankton types and set PFT flags
    ## 1: true, 0: false
    for i in range(model.n_pft):
        ## first n_phytoplankton are phytoplankton
        if i < model.n_phytoplankton:
                model.PFT_P[i] = 1
        if i >= model.n_phytoplankton and i < model.n_phytoplankton + model.n_zooplankton:
                model.PFT_Z[i] = 1
        if i >= model.n_phytoplankton + model.n_zooplankton:
                model.PFT_F[i] = 1

    ## set boolean flags
    model.PFT_P_bool = (model.PFT_P == 1)
    model.PFT_Z_bool = (model.PFT_Z == 1)
    model.PFT_F_bool = (model.PFT_F == 1)

    model._init_foram_types()

    model._init_array()
    model._assign_plank_par()
