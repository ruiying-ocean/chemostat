import numpy as np
import tomllib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Chemostat:
    """
    A theoretical size-based model of plankton in a chemostat, adapted from Grigoratou et al. 2019 BG.
    """
    
    def __init__(self,temp,nutrient):  
        self.T = temp+273.15   # ambient water temperature in Kelvin
        self.source_N = nutrient  # Source Nutrients Concentration {mmolNm^-3} supplying to the system
        


    def load_ecoconfig(self, file):
        "read config from file, set related parameters, and initialise arrays"
        with open(file, "rb") as f:
            data = tomllib.load(f)

        # set hyper config
        for name, value in data.get('parameters', {}).items():
            setattr(self, name, value)

        # set plankton structure
        for name, value in data.get('settings', {}).items():
            setattr(self, name, value)

        # per-morphotype FORAMECOGEM calibration (.zoo file row per type)
        self.foram_morphotype = data.get('foram_morphotype', {})

        # solver options (optional section; fall back to LSODA defaults)
        self.solver_opts = {
            'method': 'LSODA',
            'rtol': 1e-3,
            'atol': 1e-9,
            'max_step': 28.0,
            'first_step': 0.01,
            'min_step': 0.001,
            **data.get('solver', {}),
        }

        self.n_phytoplankton = len(self.esd_phytoplankton)
        self.n_zooplankton = len(self.esd_zooplankton)
        self.n_foram = len(self.esd_foram)

        self.diameter = np.concatenate([self.esd_phytoplankton,self.esd_zooplankton, self.esd_foram])
        

        self.n_pft = self.n_phytoplankton + self.n_zooplankton + self.n_foram
        
        self.PFT_P=np.zeros(self.n_pft)
        self.PFT_Z=np.zeros(self.n_pft)
        self.PFT_F=np.zeros(self.n_pft)
        self.PFT_P_bool = np.zeros(self.n_pft)
        self.PFT_Z_bool = np.zeros(self.n_pft)
        self.PFT_F_bool = np.zeros(self.n_pft)

        
        ## loop over plankton types and set PFT flags
        ## 1: true, 0: false
        for i in range(self.n_pft):
            ## first n_phytoplankton are phytoplankton
            if i < self.n_phytoplankton:
                    self.PFT_P[i] = 1
            if i >= self.n_phytoplankton and i < self.n_phytoplankton + self.n_zooplankton:
                    self.PFT_Z[i] = 1
            if i >= self.n_phytoplankton + self.n_zooplankton:
                    self.PFT_F[i] = 1

        ## set boolean flags
        self.PFT_P_bool = (self.PFT_P == 1)
        self.PFT_Z_bool = (self.PFT_Z == 1)
        self.PFT_F_bool = (self.PFT_F == 1)

        self._init_foram_types()

        self._init_array()
        self._assign_plank_par()

    def _init_foram_types(self):
        """Build per-pft foram morphotype masks (spinose / symbiont-bearing).

        Each foram entry has a type tag: 'bn' (barren non-spinose), 'bs' (barren spinose),
        'sn' (symbiont non-spinose), 'ss' (symbiont spinose). Tags drive the spine
        prey-selection kernel and the symbiont autotrophic uptake added below.
        """
        # backward compat: if not specified, treat all forams as legacy 'bn'.
        if not hasattr(self, 'type_foram'):
            self.type_foram = ['bn'] * self.n_foram
        if len(self.type_foram) != self.n_foram:
            raise ValueError(
                f'type_foram has {len(self.type_foram)} entries but esd_foram has {self.n_foram}'
            )
        valid = {'bn', 'bs', 'sn', 'ss'}
        bad = [t for t in self.type_foram if t not in valid]
        if bad:
            raise ValueError(f'unknown foram type(s): {bad}. valid: {sorted(valid)}')

        self.foram_type = np.array([''] * self.n_pft, dtype=object)
        foram_idx = np.flatnonzero(self.PFT_F_bool)
        for k, j in enumerate(foram_idx):
            self.foram_type[j] = self.type_foram[k]

        self.is_spinose = np.array([t in ('bs', 'ss') for t in self.foram_type], dtype=bool)
        self.is_symbiont = np.array([t in ('sn', 'ss') for t in self.foram_type], dtype=bool)
        # phytoplankton + symbiont-bearing forams take up N
        self.can_uptake_bool = self.PFT_P_bool | self.is_symbiont

        # per-pft FORAMECOGEM calibration arrays (col 02, 03, 06, 09, 10 of FORAMECOGEM.zoo).
        # Defaults reproduce the legacy heterotroph-only foram: herbivory only, ns=1, no spine cost.
        self.foram_ns        = np.ones(self.n_pft)        # prey-switching exponent (col 06)
        self.foram_growthc   = np.ones(self.n_pft)        # heterotrophy growth-cost factor (col 09)
        self.foram_calc_resp = np.zeros(self.n_pft)       # extra calcite respiration (col 10)
        self.is_herbivore    = np.zeros(self.n_pft, dtype=bool)
        self.is_carnivore    = np.zeros(self.n_pft, dtype=bool)

        for j in foram_idx:
            t = self.foram_type[j]
            cfg = self.foram_morphotype.get(t, {})
            self.foram_ns[j]        = cfg.get('ns', 1.0)
            self.foram_growthc[j]   = cfg.get('growthcost_factor', 1.0)
            self.foram_calc_resp[j] = cfg.get('calcite_respir', 0.0)
            self.is_herbivore[j]    = bool(cfg.get('herbivory', True))
            self.is_carnivore[j]    = bool(cfg.get('carnivory', False))

    def _init_array(self):
        "initialise arrays for storing parameters and output data"
        self.Qmin = np.zeros(self.n_pft)
        self.Qmax = np.zeros(self.n_pft)
        self.deltaQ = np.zeros(self.n_pft)

        self.pcmax = np.zeros(self.n_pft)
        self.pcmax_a=np.zeros(self.n_pft)
        self.mumax = np.zeros(self.n_pft)
        self.Vmax = np.zeros(self.n_pft)
        self.kNO3 = np.zeros(self.n_pft)

        self.kN = np.zeros(self.n_pft)
        self.Gmax=np.zeros(self.n_pft)

        self.m=np.zeros(self.n_pft)      # mortality
        self.resp = np.ones(self.n_pft)  # respiration
        self.Vol = np.zeros(self.n_pft)  # biovolume
        self.f = np.zeros([self.n_pft, self.n_pft]) # Grazing matrix

        self.N0 = np.ones(1) * self.init_N  # nutrient array at t0
        self.B0 = np.ones(self.n_pft) * self.init_B  # biomass array at t0

        self.output_t = np.zeros([self.save_freq, 1])
        self.output_B = np.zeros([self.save_freq, self.n_pft])  # output array for plankton biomass through time
        self.output_N = np.zeros([self.save_freq, 1])  # output array for nutrient through time
        self.graze_dt = np.zeros([self.n_pft, self.n_pft]) ## grazing term in each time step
        
        

    def _cal_platability(self, size_pred, size_prey, sigma, theta_opt):
        "calculate prey palatability based on equation A21, Ward et al 2012"

        size_ratio = size_pred / size_prey
        numerator = np.log(size_ratio / theta_opt) ** 2 * -1
        denominator = 2 * sigma ** 2
        return np.exp(numerator / denominator)

    def _cal_pcmax_a(self, diameter):
        if diameter < 0.8:
            return 1.0
        elif diameter < 1.8 and diameter >= 0.8:
            return 1.4
        elif diameter < 5.6 and diameter >= 1.8:
            return 2.1
        elif diameter >= 5.6:
            return 3.8


    def _cal_mumax(self, muinf, Vmax, Qmax, Qmin, deltaQ):
        "unit: day-1"
        diviend = muinf * Vmax * deltaQ
        divisor = Vmax * Qmax + muinf * Qmin * deltaQ
        
        return diviend / divisor

    def _cal_kN(self, muinf, kNO3, Vmax, Qmax, Qmin, deltaQ):
        "unit: mmol N m-3"
        diviend = muinf * kNO3 * Qmin * deltaQ
        divisor = Vmax * Qmax + muinf * Qmin * deltaQ
        
        return diviend / divisor

    def _cal_allom_trait(self, a, b, V):
        return a * V ** b

    

    def _assign_plank_par(self):
        """
        set ecophysiological parameters for each plankton type
        """        

        # volume plankton cell {micrometers^3}
        self.Vol = 4.0 / 3.0 * np.pi * (self.diameter*0.5)**3  

        cal_pcmax_a = np.vectorize(self._cal_pcmax_a)
        self.pcmax_a[0:self.n_phytoplankton] = cal_pcmax_a(self.esd_phytoplankton)
        
        ## assign values for all plankton
        ## Quota
        self.Qmax = self._cal_allom_trait(self.Qmax_a, self.Qmax_b, self.Vol)
        self.Qmin = self._cal_allom_trait(self.Qmin_a, self.Qmin_b, self.Vol)
        self.deltaQ = self.Qmax - self.Qmin
        self.Vmax = self._cal_allom_trait(self.Vmax_a, self.Vmax_b, self.Vol)
        self.Gmax = self._cal_allom_trait(self.Gmax_a, self.Gmax_b, self.Vol)
        self.kNO3 = self._cal_allom_trait(self.kNO3_a, self.kNO3_b, self.Vol)
        self.qcarbon = self._cal_allom_trait(self.qcarbon_a, self.qcarbon_b, self.Vol)
        self.qnitrogen = self.qcarbon * 106/16 ## redfield ratio
        ## pcmax = muinf in the paper
        self.pcmax = self._cal_allom_trait(self.pcmax_a, self.pcmax_b, self.Vol)

        ## group-specific values
        self.mumax= self._cal_mumax(self.pcmax,self.Vmax,self.Qmax, self.Qmin,self.deltaQ) * self.PFT_P
        self.kN = self._cal_kN(self.pcmax, self.kNO3,self.Vmax,self.Qmax,self.Qmin,self.deltaQ) * self.PFT_P
        ## set Gmax to 0 for phtyoplankton
        self.Gmax = self.Gmax * (1-self.PFT_P)

        # Symbiont-bearing forams (sn, ss) get autotrophic uptake derived from a tiny
        # "auto volume" V*scale^3 representing the symbiont biomass living in the host.
        # See ecogem_data.f90:553-586 — uptake traits use auto_volume rather than host volume,
        # and the result is multiplied by foram_auto_cost (energetic penalty of hosting).
        sym = self.is_symbiont
        if np.any(sym):
            auto_vol = self.Vol[sym] * self.foram_symbiont_esd_scale ** 3
            Qmax_s = self._cal_allom_trait(self.Qmax_a, self.Qmax_b, auto_vol)
            Qmin_s = self._cal_allom_trait(self.Qmin_a, self.Qmin_b, auto_vol)
            dQ_s   = Qmax_s - Qmin_s
            Vmax_s = self._cal_allom_trait(self.Vmax_a, self.Vmax_b, auto_vol)
            kNO3_s = self._cal_allom_trait(self.kNO3_a, self.kNO3_b, auto_vol)
            # symbiont auto_vol is tiny so its diameter falls in the smallest pcmax_a bucket
            pcmax_s = 1.0 * auto_vol ** self.pcmax_b
            self.mumax[sym] = self._cal_mumax(pcmax_s, Vmax_s, Qmax_s, Qmin_s, dQ_s) * self.foram_auto_cost
            self.kN[sym]    = self._cal_kN(pcmax_s, kNO3_s, Vmax_s, Qmax_s, Qmin_s, dQ_s)

        self.m = self.mp * np.ones(self.n_pft)
        self.resp = self.resp_p * np.ones(self.n_pft)

        for i in range(self.n_pft):
            if self.PFT_F_bool[i]:
                # universal foram costs from FORAMECOGEM.zoo cols 07 (mortality protection)
                # and 09 (heterotrophy growth-cost factor). cal_pm replaces the old cal_cost
                # mechanism (now growthcost_factor is per-morphotype).
                self.Gmax[i] = self.Gmax[i] * self.foram_growthc[i]
                self.m[i]    = self.m[i] * self.cal_pm
                # extra respiration cost for building calcite test (col 10) — spinose pay more
                self.resp[i] = self.resp[i] + self.foram_calc_resp[i]
            # symbiont-bearing forams pay an additional heterotrophic cost
            # (ecogem_data.f90:558 — heterotrophy *= symbiont_hetero_cost for sn/ss)
            if self.is_symbiont[i]:
                self.Gmax[i] = self.Gmax[i] * self.foram_hetero_cost

        # prey preference matrix
        for jpred in range(self.n_pft):
            if self.PFT_P_bool[jpred]:
                continue

            if self.PFT_F_bool[jpred]:
                # ECOGEM kernel is diameter-based (Ward et al. 2018): theta_opt and sigma
                # are in log-diameter-ratio units. foram_spine_scale is a diameter scaling
                # (spines extend predator diameter by spine_scale), so we apply spine_scale^1
                # — the geometrically correct multiplier for a diameter-based kernel.
                # NOTE: FORAMECOGEM's Fortran code uses spine_scale^3, which is wrong:
                # it applies a volume scaling to diameter-ratio parameters. We deliberately
                # diverge from that bug.
                sigma_pred = self.sigma_f
                theta_pred = self.theta_opt
                if self.is_spinose[jpred]:
                    sigma_pred = sigma_pred * self.foram_spine_scale
                    theta_pred = theta_pred * self.foram_spine_scale
            else:
                # zooplankton predator: keep Grigoratou et al. kernel width
                sigma_pred = self.sigma_z
                theta_pred = self.theta_opt

            for jprey in range(self.n_pft):
                # Diameter-based kernel (ECOGEM / Ward et al. 2018): predator-prey
                # palatability is a Gaussian on log(diameter ratio).
                self.f[jprey, jpred] = self._cal_platability(
                    self.diameter[jpred], self.diameter[jprey], sigma_pred, theta_pred,
                )

                # zoo predator x foram prey: scale by foram palatability (cal_pg)
                if self.PFT_Z_bool[jpred] and self.PFT_F_bool[jprey]:
                    self.f[jprey, jpred] *= self.cal_pg

            if self.PFT_F_bool[jpred]:
                # apply per-morphotype feeding mode (FORAMECOGEM.zoo cols 02, 03):
                #   herbivory=t -> can eat phyto; carnivory=t -> can eat zoo
                # ss is exclusively carnivorous (herbivory=f, carnivory=t).
                if not self.is_herbivore[jpred]:
                    self.f[self.PFT_P_bool, jpred] = 0.0
                if not self.is_carnivore[jpred]:
                    self.f[self.PFT_Z_bool, jpred] = 0.0
                # forams never eat forams (ecogem_data.f90:653)
                self.f[self.PFT_F_bool, jpred] = 0.0

            



        if not self.cannibalism:
            for i in range(self.n_pft):
                self.f[i,i] = 0.0

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

    def _phyto_uptake(self, N, B, gamma_T):
        """Monod nitrate uptake for phytoplankton and symbiont-bearing forams (sn, ss).

        Phytoplankton use the depth-averaged light term gamma_l. Symbiont-bearing
        forams live in the sunlit photic zone, so they use gamma_l_symbiont
        (default 1.0 = surface light, no depth averaging).
        """
        out = np.zeros(self.n_pft)
        light = np.where(self.is_symbiont, self.gamma_l_symbiont, self.gamma_l)
        u = self.can_uptake_bool
        out[u] = light[u] * gamma_T * self.mumax[u] * N / (self.kN[u] + N) * B[u]
        return out

    def _zoo_graze_one(self, i, B, gamma_T, dBdt):
        "Accumulate grazing contribution of zooplankton predator i into dBdt (size-selective with prey switching)."
        f_i = self.f[:, i]
        F = np.sum(f_i * B)
        if F <= 0.0:
            return
        PR = 1.0 - np.exp(self.lamdaprey * F)
        denom = np.sum(f_i * B ** 2)

        # prey-switching weights: same value for every prey within a category
        pref = np.zeros(self.n_pft)
        pref[self.PFT_P_bool] = np.sum(f_i[self.PFT_P_bool] * B[self.PFT_P_bool] ** 2) / denom
        pref[self.PFT_Z_bool] = np.sum(f_i[self.PFT_Z_bool] * B[self.PFT_Z_bool] ** 2) / denom
        pref[self.PFT_F_bool] = np.sum(f_i[self.PFT_F_bool] * B[self.PFT_F_bool] ** 2) / denom
        if np.any(pref > 1.0):
            raise ValueError('pref>1, this should not happen')

        # foram prey ignore the refuge term
        pr_eff = np.where(self.PFT_F_bool, 1.0, PR)

        graze = self.Gmax[i] * gamma_T * f_i * B / (F + self.knprey) * pr_eff * pref
        graze = np.maximum(graze, 0.0)

        dBdt[i] += np.sum(graze) * B[i] * self.lamda
        dBdt -= graze * B[i]

    def _foram_graze_one(self, i, B, gamma_T, dBdt):
        """Accumulate grazing contribution of foram predator i into dBdt.

        Generalised to per-morphotype prey-switching exponent ns (FORAMECOGEM.zoo col 06):
            graze_j = total_rate * (f_j * B_j)^ns / sum_k (f_k * B_k)^ns
        where total_rate is the Holling type II response on linearly weighted prey.
        The kernel sits inside the power, as in ECOGEM's food_ns (ecogem_box.f90 grazing).
        ns=1 reproduces the legacy (no-switching) formulation; ns=5 (bs, ss) concentrates
        feeding on the best-matched, most abundant prey within the allowed prey set.
        """
        f_i = self.f[:, i]
        F_total = np.sum(f_i * B)
        if F_total <= 0.0:
            return
        PR = 1.0 - np.exp(self.lamdaprey * F_total)

        ns = self.foram_ns[i]
        # B can dip slightly negative during ODE internal steps; (-x)**ns with float ns
        # returns NaN. Clip to zero before raising to the switching exponent.
        food_ns = (f_i * np.maximum(B, 0.0)) ** ns
        W = np.sum(food_ns)
        if W <= 0.0:
            return

        total_rate = self.Gmax[i] * gamma_T * F_total / (F_total + self.knprey) * PR
        graze = total_rate * food_ns / W
        graze = np.maximum(graze, 0.0)
        dBdt[i] += np.sum(graze) * B[i] * self.lamda
        dBdt -= graze * B[i]

    def diff_eqn(self, y, t):
        """ODE RHS fed to odeint. y = [N, B_0, ..., B_{n_pft-1}]; returns dy/dt."""
        N = y[0]
        B = y[1:self.n_pft + 1]

        gamma_T = np.exp(self.R * (self.T - self.T_ref))

        Nuptake = self._phyto_uptake(N, B, gamma_T)

        dBdt = np.zeros(self.n_pft)
        for i in np.flatnonzero(self.PFT_Z_bool):
            self._zoo_graze_one(i, B, gamma_T, dBdt)
        for i in np.flatnonzero(self.PFT_F_bool):
            self._foram_graze_one(i, B, gamma_T, dBdt)

        dBdt += Nuptake - B * self.m - self.K * B - B * self.resp * gamma_T
        dNdt = self.K * (self.source_N - N) - np.sum(Nuptake)

        dNdt = np.maximum(dNdt, 0)
        ## foram can't lose biomass (kept for parity with the original code)
        if not self.foram_neg_dBdt:
            dBdt[self.PFT_F_bool] = np.maximum(dBdt[self.PFT_F_bool], 1E-99)

        return np.append(dNdt, dBdt)

    def plot(self, sum_PFT=True, save_path=None):
        ### plot biomass, nutrient, size distribution
        plt.style.use('ggplot')
        fig, axs = plt.subplots(2,2, figsize=(8, 8), sharex=True, sharey=False, tight_layout=True)

        ## Set shared x axis
        fig.supxlabel('Time (days)')


        if sum_PFT:
            axs[0,0].plot(self.output_t, np.sum(self.output_B[:,self.PFT_P_bool], axis=1), label='Phytoplankton')
            axs[0,0].plot(self.output_t, np.sum(self.output_B[:,self.PFT_Z_bool], axis=1), label='Zooplankton')
            axs[0,0].plot(self.output_t, np.sum(self.output_B[:,self.PFT_F_bool], axis=1), label='Foraminifera')
            axs[0,0].legend()
        else:
            axs[0,0].plot(self.output_t, self.output_B[:,self.PFT_P_bool])
            axs[0,0].plot(self.output_t, self.output_B[:,self.PFT_Z_bool])
            axs[0,0].plot(self.output_t, self.output_B[:,self.PFT_F_bool])

        axs[0,0].set_title('Biomass')
        axs[0,0].set_ylabel('Biomass (mmol N m$^{-3}$)')

        axs[0,1].plot(self.output_t, self.output_N, linewidth=1)
        axs[0,1].set_title('Nutrient')
        axs[0,1].set_ylabel('Nutrient (mmol N m$^{-3}$)')

        axs[1,0].plot(self.output_t, self.output_size, linewidth=1)
        axs[1,0].set_title('Community Size Mean')
        axs[1,0].set_ylabel('Size (μm)')

        axs[1,1].plot(self.output_t, self.output_pft_richness, linewidth=1)
        axs[1,1].set_title('PFT Richness')
        axs[1,1].set_ylabel('Number of PFTs')

        if save_path is not None:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()
        return axs
