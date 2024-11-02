import numpy as np
import tomllib
from ode import ODE

class Chemostat(ODE):
    """
    A theoretical size-based model of plankton in a chemostat, adapted from Grigoratou et al. 2019 BG

    This class is a subclass of ODE. It calculates plankton parameters and inherits ODE's methods for solving the system of ODEs.
    """
    
    def __init__(self,temp,nutrient):  
        self.T = temp+273.15   # ambient water temperature in Kelvin
        self.source_N = nutrient  # Source Nutrients Concentration {mmolNm^-3} supplying to the system
        


    def load_ecoconfig(self, file):
        "read config from file, set related parameters, and initialise arrays"
        with open(file, "rb") as f:
            data = tomllib.load(f)

        # set hyper config
        for name, value in data.get('Hyper_Parameters', {}).items():
            setattr(self, name, value)

        # set plankton structure
        for name, value in data.get('Plankton', {}).items():
            setattr(self, name, value)            

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
                       
            

        self._init_array()
        self._assign_plank_par()

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
        ## pcmax = muinf in the paper
        self.pcmax = self._cal_allom_trait(self.pcmax_a, self.pcmax_b, self.Vol)

        ## group-specific values
        self.mumax= self._cal_mumax(self.pcmax,self.Vmax,self.Qmax, self.Qmin,self.deltaQ) * self.PFT_P
        self.kN = self._cal_kN(self.pcmax, self.kNO3,self.Vmax,self.Qmax,self.Qmin,self.deltaQ) * self.PFT_P
        ## set Gmax to 0 for phtyoplankton
        self.Gmax = self.Gmax * (1-self.PFT_P)
        
        self.m = self.mp * np.ones(self.n_pft)
        self.resp = self.resp_p * np.ones(self.n_pft)
        
        for i in range(self.n_pft):
            if self.PFT_F_bool[i]:
                self.Gmax[i]=self.Gmax[i] * self.cal_cost
                self.m[i]=self.m[i] * self.cal_pm
                
        # prey preference matrix        
        for jpred in range(self.n_pft):
            if self.PFT_P_bool[jpred]:
                continue
            
            for jprey in range(self.n_pft):
                # not implemented in the paper
                # if jpred == jprey:
                #     continue
                
                self.f[jprey,jpred] = self._cal_platability(self.diameter[jpred],self.diameter[jprey],self.sigma_z,self.theta_opt)

                if self.PFT_F_bool[jpred] and self.PFT_Z_bool[jprey]:
                    self.f[jprey,jpred] = 0.0

                if self.PFT_F_bool[jpred] and self.PFT_F_bool[jprey]:
                    self.f[jprey,jpred] = 0.0

                if self.PFT_F_bool[jprey]:
                    self.f[jprey,jpred] = self.f[jprey,jpred] * self.cal_pg

            

