import numpy as np
import tomllib
from ode import ODE

class Chemostat(ODE):
    """
    A theoretical size-based model of plankton in a chemostat

    Running the model:
    1. Create an instance of the model
    2. Set the model hyper-parameters
    3. Set the plankton parameters
    4. run the model and get result
    """
    
    def __init__(self,temp,nutrient):  
        self.T = temp+273.15   # ambient water temperature in Kelvin
        self.N0 = nutrient  # Source Nutrients initial Concentration {mmolNm^-3}

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

    def _platability(self, size_pred, size_prey, sigma, thita_opt):
        """
        calculate prey palatability based on equation A21, Ward et al 2012
        """
        size_ratio = size_pred / size_prey
        return np.exp(-(((np.log(size_ratio / thita_opt)) ** 2) * ((2 * sigma ** 2) ** -1)))

    def _cal_pcmax_a(self, diameter, a=0.4428, b=0.122):
        "dynamically calculate pcmax_a, fit Grigoratou example using sigmoid function"
        return a/(np.exp(-1*diameter)+b)

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

        self._set_array()
        self._set_plank_par()

    def _set_array(self):
        self.pcmax_a=np.zeros([1,self.n_pft]) # Maximum photosynthetic rate
        self.muinf = np.zeros([1,self.n_pft])
        self.Qmin = np.zeros([1,self.n_pft])  # minimum nitrogen:carbon quoata for phyto {mmolN(mmolC)^-1} 
        self.Qmax = np.zeros([1,self.n_pft])  # maximum nitrogen:carbon quoata for phyto {mmolN(mmolC)^-1}
        self.deltaQ = np.zeros([1,self.n_pft])
        self.mumax = np.zeros([1,self.n_pft])
        self.kNO3 = np.zeros([1,self.n_pft])
        self.Vmax = np.zeros([1,self.n_pft])  # maximum uptake rate of phytoplankton{mmolN(mmolC)^-1}
        self.kN = np.zeros([1,self.n_pft])  # Half- saturation concentration {mmolNm^-3}
        self.Gmax=np.zeros([1,self.n_pft])
        self.g=np.zeros([1,self.n_pft])
        self.kappa=np.zeros([1,self.n_pft])
        self.mz=np.zeros([1,self.n_pft])
        self.m=np.zeros([1,self.n_pft])
        self.gf = np.zeros([1, self.n_pft])  # grazing pressure on planktonic foraminifera
        self.Vol = np.zeros([1,self.n_pft])  # biovolume
        self.f = np.zeros([self.n_pft, self.n_pft]) # Grazing matrix

        self.N_init = np.ones([1, 1]) * 1e-4  # nutrient array at t
        self.B_init = np.ones([1, self.n_pft]) * self.init_B  # biomass array at t

        self.output_t = np.zeros([self.save_freq, 1])
        self.output_B = np.zeros([self.save_freq, self.n_pft])  # output array for plankton biomass through time
        self.output_N = np.zeros([self.save_freq, 1])  # output array for nutrient through time


    def _set_plank_par(self):
        """
        set ecophysiological parameters for each plankton type
        """

        # volume plankton cell {micrometers^3}
        self.diameter = np.reshape(self.diameter, (1, self.n_pft))
        for j in range(0, self.n_pft):
            self.Vol[0,j] = 4.0 / 3.0 * np.pi * (self.diameter[0,j]*0.5)**3  

        self.pcmax_a[0, 0:self.n_phytoplankton] = self._cal_pcmax_a(self.get_diameter(self.n_phytoplankton, "phytoplankton"))
        
        ## assign values
        for n in range(0,self.n_pft):
            ## Quota
            self.Qmax[0,n]=self.Qmax_a*self.Vol[0,n]**self.Qmax_b    
            self.Qmin[0,n]=self.Qmin_a*self.Vol[0,n]**self.Qmin_b  
            self.deltaQ[0,n]=self.Qmax[0,n]-self.Qmin[0,n]
            ## Monod function
            self.Vmax[0,n]=self.Vmax_a*self.Vol[0,n]**self.Vmax_b
            self.Gmax[0,n]=self.Gcmax_a*self.Vol[0,n]**self.Gcmax_b
            self.kNO3[0,n]=self.k_no3_a*self.Vol[0,n]**self.k_no3_b
            self.muinf[0,n]=self.pcmax_a[0,n]*self.Vol[0,n]**self.pcmax_b            
            
        for n in range(0,self.n_pft):
            if self.PFT_P[n]:
                self.mumax[0,n]=self.muinf[0,n]*self.Vmax[0,n]*self.deltaQ[0,n]/(self.Vmax[0,n]*self.Qmax[0,n]+self.muinf[0,n]*self.Qmin[0,n]*self.deltaQ[0,n])
                self.kN[0,n]=self.muinf[0,n]*self.kNO3[0,n]*self.Qmin[0,n]*self.deltaQ[0,n]/(self.Vmax[0,n]*self.Qmax[0,n]+self.muinf[0,n]*self.Qmin[0,n]*self.deltaQ[0,n])
                self.kappa[0,n]=self.K
                self.m[0,n]=self.mp
                self.gf[0,n] = 1.0
                
            elif self.PFT_Z[n]:
                self.g[0,n]=self.Gmax[0,n]
                self.kappa[0,n]=self.K
                self.gf[0,n] = 1.0
                self.mz[0,n] = self.mp 
                self.m[0,n]= self.mz[0,n]# fixed grazing rate

            elif self.PFT_F[n]:
                self.g[0,n]=self.Gmax[0,n] *0.5 # * diameter value (ranges from 0 to 1 for energetic loss)
                self.gf[0,n] = 1.0 # grazing pressure on forams, ranges from 0 to 1 for different predation pressure on forams
                self.kappa[0,n]=self.K
                self.mz[0,n] = self.mp * 0.81# * diameter value (ranges from 0 to 1 for protection from background mortality)
                self.m[0,n]= self.mz[0,n]
                
        # prey preference
        for jpred in range(0,self.n_pft):
            ## zooplankton
            if self.PFT_Z[jpred]:
                for jprey in range(0,self.n_pft):
                    self.f[jprey,jpred] = self._platability(self.diameter[0,jpred],self.diameter[0,jprey],self.sigma_z,self.thita_opt)
            ## in this version, forams only feed on phytoplankton
            if self.PFT_F[jpred]:
                for jprey in range(0,self.n_phytoplankton):
                    self.f[jprey,jpred] = self._platability(self.diameter[0,jpred],self.diameter[0,jprey],self.sigma_f,self.thita_opt)
                        
        self.graze_dt=np.zeros([self.n_pft,self.n_pft])

