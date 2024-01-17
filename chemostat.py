import numpy as np
import tomllib
from ode import ODE

class Chemostat(ODE):
    """
    A theoretical size-based model of plankton in a chemostat, adapted from Grigoratou et al. 2019 BG
    """
    
    def __init__(self,temp,nutrient):  
        self.T = temp+273.15   # ambient water temperature in Kelvin
        self.source_N = nutrient  # Source Nutrients Concentration {mmolNm^-3} supplying to the system

    def get_diameter(self, n, plankton):
        """
        generate n diameters for a given plankton type
        fitted from Grigoratou et al. 2019 BG
        zooplankton sampled from 0-1900 um
        phytoplankton sampled from 0-190 um
        """
        coef_a = {
            "phytoplankton": 0.625,
            "zooplankton": 6.25,
            "foraminifera": 6.25,
        }
        a = coef_a[plankton]
        b = 0.23
        index = np.linspace(0, 25, n)
        return a * np.exp(b * index)

    def _cal_platability(self, size_pred, size_prey, sigma, thita_opt):
        "calculate prey palatability based on equation A21, Ward et al 2012"

        size_ratio = size_pred / size_prey
        return np.exp(-(((np.log(size_ratio / thita_opt)) ** 2) * ((2 * sigma ** 2) ** -1)))

    def _cal_pcmax_a(self, diameter):
        # < 0.8 => 1.0
        # < 1.8 => 1.4
        # < 5.6 => 2.1
        # > 5.6 => 3.8
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

    def load_config(self, file):
        with open(file, "rb") as f:
            data = tomllib.load(f)

        # set hyper config
        for name, value in data.get('Hyper_Parameters', {}).items():
            setattr(self, name, value)

        # set plankton config
        for name, value in data.get('Plankton', {}).items():
            setattr(self, name, value)
            
        self.n_pft = self.n_phytoplankton + self.n_zooplankton + self.n_foram

        self.diameter = np.concatenate(
            (
                self.get_diameter(self.n_phytoplankton, "phytoplankton"),
                self.get_diameter(self.n_zooplankton, "zooplankton"),
                self.get_diameter(self.n_foram, "foraminifera"),
            )
        )
        
        self.PFT_P=np.zeros([self.n_pft],dtype=bool)
        self.PFT_Z=np.zeros([self.n_pft],dtype=bool)
        self.PFT_F=np.zeros([self.n_pft],dtype=bool)
        
        self.PFT_P[0:self.n_phytoplankton]=True
        self.PFT_Z[self.n_phytoplankton:self.n_phytoplankton+self.n_zooplankton]=True
        self.PFT_F[self.n_phytoplankton+self.n_zooplankton:self.n_pft]=True

        self._init_array()
        self._set_plank_par()

    def _init_array(self):
        self.pcmax = np.zeros([1,self.n_pft])
        self.pcmax_a=np.zeros([1,self.n_pft])
        self.Qmin = np.zeros([1,self.n_pft])
        self.Qmax = np.zeros([1,self.n_pft])
        self.deltaQ = np.zeros([1,self.n_pft])
        self.mumax = np.zeros([1,self.n_pft])
        self.kNO3 = np.zeros([1,self.n_pft])
        self.Vmax = np.zeros([1,self.n_pft])
        self.kN = np.zeros([1,self.n_pft])
        self.Gmax=np.zeros([1,self.n_pft])
        
        self.g=np.zeros([1,self.n_pft])
        self.kappa=np.ones([1,self.n_pft]) * self.K
        self.mz=np.zeros([1,self.n_pft])
        self.m=np.zeros([1,self.n_pft])
        self.Vol = np.zeros([1,self.n_pft])  # biovolume
        self.f = np.zeros([self.n_pft, self.n_pft]) # Grazing matrix

        self.N0 = np.ones([1, 1]) * self.init_N  # nutrient array at t0
        self.B0 = np.ones([1, self.n_pft]) * self.init_B  # biomass array at t0

        self.output_t = np.zeros([self.save_freq, 1])
        self.output_B = np.zeros([self.save_freq, self.n_pft])  # output array for plankton biomass through time
        self.output_N = np.zeros([self.save_freq, 1])  # output array for nutrient through time


    def _set_plank_par(self):
        """
        set ecophysiological parameters for each plankton type
        """

        
        self.diameter = np.reshape(self.diameter, (1, self.n_pft))

        # volume plankton cell {micrometers^3}
        for j in range(0, self.n_pft):
            self.Vol[0,j] = 4.0 / 3.0 * np.pi * (self.diameter[0,j]*0.5)**3  

        cal_pcmax_a = np.vectorize(self._cal_pcmax_a)
        self.pcmax_a[0, 0:self.n_phytoplankton] = cal_pcmax_a(self.get_diameter(self.n_phytoplankton, "phytoplankton"))
        
        ## assign values for all plankton
        for n in range(0,self.n_pft):
            ## Quota
            self.Qmax[0,n] = self._cal_allom_trait(self.Qmax_a,self.Qmax_b,self.Vol[0,n])
            self.Qmin[0,n] = self._cal_allom_trait(self.Qmin_a,self.Qmin_b,self.Vol[0,n])
            self.deltaQ[0,n] = self.Qmax[0,n] - self.Qmin[0,n]
            self.Vmax[0,n]= self._cal_allom_trait(self.Vmax_a,self.Vmax_b,self.Vol[0,n])
            self.Gmax[0,n]= self._cal_allom_trait(self.Gmax_a,self.Gmax_b,self.Vol[0,n])
            self.kNO3[0,n]= self._cal_allom_trait(self.kNO3_a,self.kNO3_b,self.Vol[0,n])
            ## pcmax = muinf in the paper
            self.pcmax[0,n]= self._cal_allom_trait(self.pcmax_a[0,n],self.pcmax_b,self.Vol[0,n])

        ## group-specific values
        for n in range(0,self.n_pft):
            if self.PFT_P[n]:
                self.mumax[0,n]= self._cal_mumax(self.pcmax[0,n],self.Vmax[0,n],self.Qmax[0,n], self.Qmin[0,n],self.deltaQ[0,n])
                self.kN[0,n]= self._cal_kN(self.pcmax[0,n], self.kNO3[0,n],self.Vmax[0,n],self.Qmax[0,n],self.Qmin[0,n],self.deltaQ[0,n])
                self.m[0,n]=self.mp
                
            elif self.PFT_Z[n]:
                self.m[0,n]= self._cal_allom_trait(self.mz_a,self.mz_b,self.Vol[0,n])

            elif self.PFT_F[n]:
                self.Gmax[0,n]=self.Gmax[0,n] *0.5
                self.m[0,n] = self._cal_allom_trait(self.mz_a,self.mz_b,self.Vol[0,n]) * 0.81
                
        # prey preference
        for jpred in range(0,self.n_pft):
            ## zooplankton
            if self.PFT_Z[jpred]:
                for jprey in range(0,self.n_pft):
                    self.f[jprey,jpred] = self._cal_platability(self.diameter[0,jpred],self.diameter[0,jprey],self.sigma_z,self.thita_opt)
            ## in this version, forams only feed on phytoplankton
            if self.PFT_F[jpred]:
                for jprey in range(0,self.n_phytoplankton):
                    self.f[jprey,jpred] = self._cal_platability(self.diameter[0,jpred],self.diameter[0,jprey],self.sigma_f,self.thita_opt)
                        
        self.graze_dt=np.zeros([self.n_pft,self.n_pft])

