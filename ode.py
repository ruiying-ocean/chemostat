import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class ODE:
    def run(self):
        
        ## set time, init condition
        t = np.linspace(1.0, self.nday, self.save_freq)
        y0 = np.append(self.N0, self.B0)

        ### call solver ###
        ## diff_eqn is the function describing the system, in the form of f(y,t);
        ## y could be multi-dimension, but t must be 1-dimension
        print('>>> running ODE solver')
        sol = odeint(self.diff_eqn, y0, t, full_output=False, rtol=1e-3, atol=1e-9, hmax=28.0, h0=0.01, hmin=0.001)

        # return solver output to output arrays
        self.output_t = t
        ## first index is nutrient
        self.output_N[:, 0] = sol[:, 0]
        ## the others are plankton
        self.output_B[:, :] = sol[:, 1:self.n_pft+1]

        ## weighted mean diameter
        ## calculate biomass proportion of each PFT
        self.output_prop = self.output_B / np.sum(self.output_B, axis=1)[:, None]
        ## calculate weighted mean diameter
        self.output_size = np.sum(self.output_prop * self.diameter, axis=1)

        ## calculate diversity
        self.output_pft_richness  = np.sum(self.output_B > self.qnitrogen, axis=1)

    def diff_eqn(self, y, t):
        """differential equation fed to odeint solver
        y: state variable (including both N and B)
        t: time
        """

        ## state variable
        N = np.zeros(1)
        B = np.zeros(self.n_pft)
        Nuptake = np.zeros(self.n_pft)
        dBdt = np.zeros(self.n_pft)
        dNdt = np.zeros(1)
        
        ## initial condition
        N = y[0]
        B = y[1:self.n_pft + 1]

        self.gamma_T = np.exp(self.R * (self.T - self.T_ref))


        for i in range(self.n_pft):                    
            if self.PFT_P_bool[i]: # phyto                
                Nuptake[i]=self.gamma_l* self.gamma_T * self.mumax[i]*N/(self.kN[i]+N)*B[i]                             
                
            elif self.PFT_Z_bool[i]: # zoo                
                F=np.sum(self.f[:,i]*B) # availability of prey                            
                if F>0.0:
                    PR= 1.0-np.exp(self.lamdaprey*F)  
                                        
                    for jprey in range(self.n_pft): # loop over prey                        
                        if self.PFT_P_bool[jprey]:
                            pref=np.sum(self.f[self.PFT_P_bool,i]*B[self.PFT_P_bool]**2) / np.sum(self.f[:,i]*B**2)
                        elif self.PFT_Z_bool[jprey]:
                            pref=np.sum(self.f[self.PFT_Z_bool,i]*B[self.PFT_Z_bool]**2) / np.sum(self.f[:,i]*B**2)
                        elif self.PFT_F_bool[jprey]:
                            PR = 1.0 #no prey refuge
                            pref=np.sum(self.f[self.PFT_F_bool,i]*B[self.PFT_F_bool]**2) / np.sum(self.f[:,i]*B**2) 
                        if pref>1.0:
                            raise ValueError('pref>1, this should not happen')                            

                             
                        graze=self.Gmax[i]*self.gamma_T*(self.f[jprey,i]*B[jprey]/(F+self.knprey)) * PR * pref

                        graze = np.max(graze, 0)
                                                                
                        dBdt[i]=dBdt[i]+(graze*B[i]*self.lamda)
                            
                        dBdt[jprey]=dBdt[jprey]-(graze*B[i])                     
                        
            elif self.PFT_F_bool[i]: # forams
                F=np.sum(self.f[:,i]*B) # availability of prey
                
                if F>0.0:
                    PR= 1.0-np.exp(self.lamdaprey*F)                    
                    for jprey in range(self.n_phytoplankton): # loop over prey
                        graze = self.Gmax[i]*self.gamma_T*(self.f[jprey,i]*B[jprey]/(F+self.knprey)) * PR 
                        graze = np.max(graze, 0)
                        dBdt[i]=dBdt[i]+(graze*B[i]*self.lamda)
                        dBdt[jprey]=dBdt[jprey]-(graze*B[i])


        self.Nuptake=Nuptake      


        dBdt=dBdt+Nuptake-(B*self.m)-(self.K*B) -(B*self.resp*self.gamma_T)
        dNdt=self.K*(self.source_N-N) - np.sum(Nuptake)


        dNdt = np.max(dNdt, 0)
        Nuptake = 0 if dNdt < 0 else Nuptake
        ## foram can't have lose biomass (I know it is radiculouse, but that's the original code)
        if not self.foram_neg_dBdt:            
            dBdt[self.PFT_F_bool] = np.maximum(dBdt[self.PFT_F_bool], 1E-99)
        

        return np.append(dNdt,dBdt)



    def plot(self, sum_PFT=True):
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

        plt.show()
        return axs
