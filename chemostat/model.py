"""Combine ecological processes and integrate the chemostat model."""

import numpy as np
from scipy.integrate import solve_ivp

from . import components, config, traits


class Chemostat:
    """
    A theoretical size-based model of plankton in a chemostat, adapted from Grigoratou et al. 2019 BG.
    """
    # calcification as a share of foram max growth (not in FORAMECOGEM; 0 = off).
    # Spinose forams pay foram_spinose_growth_cost in place of foram_growth_cost.
    foram_growth_cost = 0.0
    foram_spinose_growth_cost = 0.0
    # foram Q10 by symbiont trait (not in FORAMECOGEM); None = shared curve exp(R*(T-T_ref))
    foram_Q10 = None
    foram_symbiont_Q10 = None

    def __init__(self, temp, nutrient):
        self.T = temp + 273.15   # ambient water temperature in Kelvin
        self.source_N = nutrient  # Source Nutrients Concentration {mmolNm^-3} supplying to the system
        self.processes = components.DEFAULT_PROCESSES.copy()

    def load_ecoconfig(self, file):
        """Load model parameters, traits, and initial conditions from TOML."""
        config.load_ecoconfig(self, file)

    def _init_foram_types(self):
        traits.init_foram_types(self)

    def _init_array(self):
        traits.init_arrays(self)

    def _assign_plank_par(self):
        traits.assign_plankton_traits(self)

    # Keep the existing trait helper names available to scripts and notebooks.
    _cal_platability = staticmethod(traits.palatability)
    _cal_pcmax_a = staticmethod(traits.photosynthesis_size_factor)
    _cal_mumax = staticmethod(traits.maximum_growth)
    _cal_kN = staticmethod(traits.nutrient_half_saturation)
    _cal_allom_trait = staticmethod(traits.allometric_trait)

    def run(self):

        ## set time, init condition
        t = np.linspace(1.0, self.nday, self.save_freq)
        y0 = np.append(self.N0, self.B0)

        ### call solver ###
        ## diff_eqn keeps the legacy f(y, t) signature; solve_ivp expects f(t, y), so wrap it.
        # min_step is LSODA-only — drop it for other methods so solve_ivp doesn't raise.
        kwargs = dict(self.solver_opts)
        if kwargs.get('method') != 'LSODA':
            kwargs.pop('min_step', None)
        print(f">>> running ODE solver (solve_ivp, {kwargs['method']})")
        sol = solve_ivp(
            lambda t, y: self.diff_eqn(y, t),
            (t[0], t[-1]),
            y0,
            t_eval=t,
            **kwargs,
        )
        if not sol.success:
            raise RuntimeError(f'ODE solver failed: {sol.message}')
        print('<<< simulation complete')
        # return solver output to output arrays; solve_ivp gives y of shape (n_states, n_t)
        self.output_t = sol.t
        ## first index is nutrient
        self.output_N[:, 0] = sol.y[0, :]
        ## the others are plankton
        self.output_B[:, :] = sol.y[1:self.n_pft+1, :].T

        ## weighted mean diameter
        ## calculate biomass proportion of each PFT
        self.output_prop = self.output_B / np.sum(self.output_B, axis=1)[:, None]
        ## calculate weighted mean diameter
        self.output_size = np.sum(self.output_prop * self.diameter, axis=1)

        ## calculate diversity
        self.output_pft_richness  = np.sum(self.output_B > self.qnitrogen, axis=1)

    def diff_eqn(self, y, t):
        """ODE RHS fed to solve_ivp. y = [N, B_0, ..., B_{n_pft-1}]; returns dy/dt."""
        # the solver can overshoot slightly below zero; rates from negative values run away
        y = np.maximum(y, 0.0)
        N = y[0]
        B = y[1:self.n_pft + 1]

        gamma_T = np.exp(self.R_pft * (self.T - self.T_ref))  # per pft

        Nuptake = self._phyto_uptake(N, B, gamma_T)

        dBdt = np.zeros(self.n_pft)
        for i in np.flatnonzero(self.PFT_Z_bool):
            self._zoo_graze_one(i, B, gamma_T[i], dBdt)
        for i in np.flatnonzero(self.PFT_F_bool):
            self._foram_graze_one(i, B, gamma_T[i], dBdt)

        dBdt += (
            Nuptake
            - self.processes["mortality"](B, self.m)
            - self.processes["dilution"](B, self.K)
            - self.processes["respiration"](B, gamma_T, self.resp)
        )
        # no clamp on dNdt: uptake -> 0 as N -> 0 keeps N >= 0, and a clamp would create nitrogen
        dNdt = self.processes["nutrient_supply"](N, self.source_N, self.K) - np.sum(Nuptake)

        return np.append(dNdt, dBdt)

    def _phyto_uptake(self, N, B, gamma_T):
        light = np.where(self.is_symbiont, self.gamma_l_symbiont, self.gamma_l)
        return self.processes["uptake"](
            N, B, gamma_T, light=light, mumax=self.mumax, kN=self.kN,
            can_uptake=self.can_uptake_bool,
        )

    def _zoo_graze_one(self, i, B, gamma_T, dBdt):
        self.processes["zooplankton_grazing"](
            i, B, gamma_T, dBdt, preference=self.f[:, i], Gmax=self.Gmax[i],
            refuge_strength=self.lamdaprey, half_saturation=self.knprey,
            assimilation=self.lamda,
            prey_groups=(self.PFT_P_bool, self.PFT_Z_bool, self.PFT_F_bool),
        )

    def _foram_graze_one(self, i, B, gamma_T, dBdt):
        self.processes["foram_grazing"](
            i, B, gamma_T, dBdt, preference=self.f[:, i], Gmax=self.Gmax[i],
            refuge_strength=self.lamdaprey, half_saturation=self.knprey,
            assimilation=self.lamda, switching_exponent=self.foram_ns[i],
        )

    def plot(self, sum_PFT=True, save_path=None):
        """Plot biomass, nutrients, mean size, and richness."""
        from .plotting import plot

        return plot(self, sum_PFT=sum_PFT, save_path=save_path)
