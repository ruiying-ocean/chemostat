"""Size-dependent traits, foram morphotypes, and prey preferences."""

import numpy as np


def palatability(size_pred, size_prey, sigma, theta_opt):
    "calculate prey palatability based on equation A21, Ward et al 2012"

    size_ratio = size_pred / size_prey
    numerator = np.log(size_ratio / theta_opt) ** 2 * -1
    denominator = 2 * sigma ** 2
    return np.exp(numerator / denominator)


def photosynthesis_size_factor(diameter):
    if diameter < 0.8:
        return 1.0
    elif diameter < 1.8 and diameter >= 0.8:
        return 1.4
    elif diameter < 5.6 and diameter >= 1.8:
        return 2.1
    elif diameter >= 5.6:
        return 3.8


def maximum_growth(muinf, Vmax, Qmax, Qmin, deltaQ):
    "unit: day-1"
    diviend = muinf * Vmax * deltaQ
    divisor = Vmax * Qmax + muinf * Qmin * deltaQ

    return diviend / divisor


def nutrient_half_saturation(muinf, kNO3, Vmax, Qmax, Qmin, deltaQ):
    "unit: mmol N m-3"
    diviend = muinf * kNO3 * Qmin * deltaQ
    divisor = Vmax * Qmax + muinf * Qmin * deltaQ

    return diviend / divisor


def allometric_trait(a, b, V):
    return a * V ** b


def init_foram_types(model):
    """Build per-pft foram morphotype masks (spinose / symbiont-bearing).

    Each foram entry has a type tag: 'bn' (barren non-spinose), 'bs' (barren spinose),
    'sn' (symbiont non-spinose), 'ss' (symbiont spinose). Tags drive the spine
    prey-selection kernel and the symbiont autotrophic uptake added below.
    """
    # backward compat: if not specified, treat all forams as legacy 'bn'.
    if not hasattr(model, 'type_foram'):
        model.type_foram = ['bn'] * model.n_foram
    if len(model.type_foram) != model.n_foram:
        raise ValueError(
            f'type_foram has {len(model.type_foram)} entries but esd_foram has {model.n_foram}'
        )
    valid = {'bn', 'bs', 'sn', 'ss'}
    bad = [t for t in model.type_foram if t not in valid]
    if bad:
        raise ValueError(f'unknown foram type(s): {bad}. valid: {sorted(valid)}')

    model.foram_type = np.array([''] * model.n_pft, dtype=object)
    foram_idx = np.flatnonzero(model.PFT_F_bool)
    for k, j in enumerate(foram_idx):
        model.foram_type[j] = model.type_foram[k]

    model.is_spinose = np.array([t in ('bs', 'ss') for t in model.foram_type], dtype=bool)
    model.is_symbiont = np.array([t in ('sn', 'ss') for t in model.foram_type], dtype=bool)
    # phytoplankton + symbiont-bearing forams take up N
    model.can_uptake_bool = model.PFT_P_bool | model.is_symbiont

    # per-pft FORAMECOGEM calibration arrays (col 02, 03, 06, 09, 10 of FORAMECOGEM.zoo).
    # Defaults reproduce the legacy heterotroph-only foram: herbivory only, ns=1, no spine cost.
    model.foram_ns        = np.ones(model.n_pft)        # prey-switching exponent (col 06)
    model.foram_growthc   = np.ones(model.n_pft)        # heterotrophy growth-cost factor (col 09)
    model.foram_calc_resp = np.zeros(model.n_pft)       # extra calcite respiration (col 10)
    model.is_herbivore    = np.zeros(model.n_pft, dtype=bool)
    model.is_carnivore    = np.zeros(model.n_pft, dtype=bool)

    for j in foram_idx:
        t = model.foram_type[j]
        cfg = model.foram_morphotype.get(t, {})
        model.foram_ns[j]        = cfg.get('ns', 1.0)
        model.foram_growthc[j]   = cfg.get('growthcost_factor', 1.0)
        model.foram_calc_resp[j] = cfg.get('calcite_respir', 0.0)
        model.is_herbivore[j]    = bool(cfg.get('herbivory', True))
        model.is_carnivore[j]    = bool(cfg.get('carnivory', False))


def init_arrays(model):
    "initialise arrays for storing parameters and output data"
    model.Qmin = np.zeros(model.n_pft)
    model.Qmax = np.zeros(model.n_pft)
    model.deltaQ = np.zeros(model.n_pft)

    model.pcmax = np.zeros(model.n_pft)
    model.pcmax_a=np.zeros(model.n_pft)
    model.mumax = np.zeros(model.n_pft)
    model.Vmax = np.zeros(model.n_pft)
    model.kNO3 = np.zeros(model.n_pft)

    model.kN = np.zeros(model.n_pft)
    model.Gmax=np.zeros(model.n_pft)

    model.m=np.zeros(model.n_pft)      # mortality
    model.resp = np.ones(model.n_pft)  # respiration
    model.Vol = np.zeros(model.n_pft)  # biovolume
    model.f = np.zeros([model.n_pft, model.n_pft]) # Grazing matrix

    model.N0 = np.ones(1) * model.init_N  # nutrient array at t0
    model.B0 = np.ones(model.n_pft) * model.init_B  # biomass array at t0

    model.output_t = np.zeros([model.save_freq, 1])
    model.output_B = np.zeros([model.save_freq, model.n_pft])  # output array for plankton biomass through time
    model.output_N = np.zeros([model.save_freq, 1])  # output array for nutrient through time
    model.graze_dt = np.zeros([model.n_pft, model.n_pft]) ## grazing term in each time step


def assign_plankton_traits(model):
    """
    set ecophysiological parameters for each plankton type
    """

    # volume plankton cell {micrometers^3}
    model.Vol = 4.0 / 3.0 * np.pi * (model.diameter*0.5)**3

    cal_pcmax_a = np.vectorize(photosynthesis_size_factor)
    model.pcmax_a[0:model.n_phytoplankton] = cal_pcmax_a(model.esd_phytoplankton)

    ## assign values for all plankton
    ## Quota
    model.Qmax = allometric_trait(model.Qmax_a, model.Qmax_b, model.Vol)
    model.Qmin = allometric_trait(model.Qmin_a, model.Qmin_b, model.Vol)
    model.deltaQ = model.Qmax - model.Qmin
    model.Vmax = allometric_trait(model.Vmax_a, model.Vmax_b, model.Vol)
    model.Gmax = allometric_trait(model.Gmax_a, model.Gmax_b, model.Vol)
    model.kNO3 = allometric_trait(model.kNO3_a, model.kNO3_b, model.Vol)
    model.qcarbon = allometric_trait(model.qcarbon_a, model.qcarbon_b, model.Vol)
    model.qnitrogen = model.qcarbon * 16 / 106  # N per cell (Redfield N:C); richness counts B > one cell per m^3
    ## pcmax = muinf in the paper
    model.pcmax = allometric_trait(model.pcmax_a, model.pcmax_b, model.Vol)

    ## group-specific values
    model.mumax= maximum_growth(model.pcmax,model.Vmax,model.Qmax, model.Qmin,model.deltaQ) * model.PFT_P
    model.kN = nutrient_half_saturation(model.pcmax, model.kNO3,model.Vmax,model.Qmax,model.Qmin,model.deltaQ) * model.PFT_P
    ## set Gmax to 0 for phtyoplankton
    model.Gmax = model.Gmax * (1-model.PFT_P)

    # Symbiont-bearing forams (sn, ss) get autotrophic uptake derived from a tiny
    # "auto volume" V*scale^3 representing the symbiont biomass living in the host.
    # See ecogem_data.f90:553-586 — uptake traits use auto_volume rather than host volume,
    # and the result is multiplied by foram_auto_cost (energetic penalty of hosting).
    sym = model.is_symbiont
    if np.any(sym):
        auto_vol = model.Vol[sym] * model.foram_symbiont_esd_scale ** 3
        Qmax_s = allometric_trait(model.Qmax_a, model.Qmax_b, auto_vol)
        Qmin_s = allometric_trait(model.Qmin_a, model.Qmin_b, auto_vol)
        dQ_s   = Qmax_s - Qmin_s
        Vmax_s = allometric_trait(model.Vmax_a, model.Vmax_b, auto_vol)
        kNO3_s = allometric_trait(model.kNO3_a, model.kNO3_b, auto_vol)
        # same size buckets as phytoplankton, at the symbiont's own diameter
        pcmax_a_s = np.array([photosynthesis_size_factor(d) for d in model.diameter[sym] * model.foram_symbiont_esd_scale])
        pcmax_s = pcmax_a_s * auto_vol ** model.pcmax_b
        model.mumax[sym] = maximum_growth(pcmax_s, Vmax_s, Qmax_s, Qmin_s, dQ_s) * model.foram_auto_cost
        model.kN[sym]    = nutrient_half_saturation(pcmax_s, kNO3_s, Vmax_s, Qmax_s, Qmin_s, dQ_s)

    model.m = model.mp * np.ones(model.n_pft)
    model.resp = model.resp_p * np.ones(model.n_pft)

    for i in range(model.n_pft):
        if model.PFT_F_bool[i]:
            # universal foram costs from FORAMECOGEM.zoo cols 07 (mortality protection)
            # and 09 (heterotrophy growth-cost factor). cal_pm replaces the old cal_cost
            # mechanism (now growthcost_factor is per-morphotype).
            model.Gmax[i] = model.Gmax[i] * model.foram_growthc[i]
            model.m[i]    = model.m[i] * model.cal_pm
            # extra respiration cost for building calcite test (col 10) — spinose pay more
            model.resp[i] = model.resp[i] + model.foram_calc_resp[i]
            # calcification cuts max growth: grazing and symbiont uptake alike
            keep = 1.0 - (model.foram_spinose_growth_cost if model.is_spinose[i] else model.foram_growth_cost)
            model.Gmax[i] *= keep
            model.mumax[i] *= keep
        # symbiont-bearing forams pay an additional heterotrophic cost
        # (ecogem_data.f90:558 — heterotrophy *= symbiont_hetero_cost for sn/ss)
        if model.is_symbiont[i]:
            model.Gmax[i] = model.Gmax[i] * model.foram_hetero_cost

    # temperature slope per pft; forams may take their own Q10 by symbiont trait
    model.R_pft = np.full(model.n_pft, float(model.R))
    for i in np.flatnonzero(model.PFT_F_bool):
        q10 = model.foram_symbiont_Q10 if model.is_symbiont[i] else model.foram_Q10
        if q10 is not None:
            model.R_pft[i] = np.log(q10) / 10.0

    # prey preference matrix
    for jpred in range(model.n_pft):
        if model.PFT_P_bool[jpred]:
            continue

        if model.PFT_F_bool[jpred]:
            # ECOGEM kernel is diameter-based (Ward et al. 2018): theta_opt and sigma
            # are in log-diameter-ratio units. foram_spine_scale is a diameter scaling
            # (spines extend predator diameter by spine_scale), so we apply spine_scale^1
            # — the geometrically correct multiplier for a diameter-based kernel.
            # NOTE: FORAMECOGEM's Fortran code uses spine_scale^3, which is wrong:
            # it applies a volume scaling to diameter-ratio parameters. We deliberately
            # diverge from that bug.
            sigma_pred = model.sigma_f
            theta_pred = model.theta_opt
            if model.is_spinose[jpred]:
                sigma_pred = sigma_pred * model.foram_spine_scale
                theta_pred = theta_pred * model.foram_spine_scale
        else:
            # zooplankton predator: keep Grigoratou et al. kernel width
            sigma_pred = model.sigma_z
            theta_pred = model.theta_opt

        for jprey in range(model.n_pft):
            # Diameter-based kernel (ECOGEM / Ward et al. 2018): predator-prey
            # palatability is a Gaussian on log(diameter ratio).
            model.f[jprey, jpred] = palatability(
                model.diameter[jpred], model.diameter[jprey], sigma_pred, theta_pred,
            )

            # zoo predator x foram prey: scale by foram palatability (cal_pg)
            if model.PFT_Z_bool[jpred] and model.PFT_F_bool[jprey]:
                model.f[jprey, jpred] *= model.cal_pg

        if model.PFT_F_bool[jpred]:
            # apply per-morphotype feeding mode (FORAMECOGEM.zoo cols 02, 03):
            #   herbivory=t -> can eat phyto; carnivory=t -> can eat zoo
            # ss is exclusively carnivorous (herbivory=f, carnivory=t).
            if not model.is_herbivore[jpred]:
                model.f[model.PFT_P_bool, jpred] = 0.0
            if not model.is_carnivore[jpred]:
                model.f[model.PFT_Z_bool, jpred] = 0.0
            # forams never eat forams (ecogem_data.f90:653)
            model.f[model.PFT_F_bool, jpred] = 0.0


    if not model.cannibalism:
        for i in range(model.n_pft):
            model.f[i,i] = 0.0
