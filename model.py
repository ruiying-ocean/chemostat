import numpy as np
import time
from scipy.integrate import odeint

class Plankton_Size_Chemostat(object):
    
    
    def __init__(self,runtime,save_freq,temp,nutrient,diameters,plankton_type):
        
        # INPUTS
        #-------------------------
        self.nday = runtime # number of days to run model over
        self.save_freq=save_freq # number of linearly spaced times to save output
        self.T = temp+273.15   # ambient water temperature {Kelvin} 273.15 for Zero oC, 283.15 for 10 oC, 303.15 for 30 oC 
        self.No = nutrient  # Deep Source Nutrients initial Concentration {mmolNm^-3}
        self.D=diameters # - diameters in um 
        # plankton type - array choosing plankton type
        #--------------------------
        
        self.n_pft=diameters.size # number of plankton
        self.PFT_P=np.zeros([self.n_pft],dtype=bool)
        self.PFT_Z=np.zeros([self.n_pft],dtype=bool)
        self.PFT_F=np.zeros([self.n_pft],dtype=bool)
        
        self.PFT_P[plankton_type[0,:]==1]=True
        self.PFT_Z[plankton_type[0,:]==2]=True
        self.PFT_F[plankton_type[0,:]==3]=True                  
        
        ### Plankton Volumes ###
        self.V = np.zeros([1,self.n_pft])  # cell/body volume
        
        for j in range(0, self.n_pft):
            self.V[0,j] = 4.0 / 3.0 * np.pi * (diameters[0,j]*0.5)**3  # volume plankton cell {micrometers^3}  
            
        ### Size Dependent Parameters (from Ward et al. 2012) ###
        self.Qmax_a=0.25
        self.Qmax_b=-0.13
        self.Qmin_a=0.07
        self.Qmin_b=-0.17
        self.Gcmax_a=21.9
        self.Gcmax_b=-0.16
        self.k_no3_a=0.17
        self.k_no3_b=0.27
        self.Vmax_a=0.51
        self.Vmax_b=-0.27
        self.pcmax_b=-0.15
        self.mz_a=0.05
        self.mz_b=-0.16
        
        ### Size-Indenpendent Parameters ###
        self.kcprey=1.0/6.625 #C TO N CONVERSION (from Ward et al. 2012)
        self.lambda_li = 0.1 #light limitation  (from Ward et al. 2014)
        
        ### Temperature- Arrenhius-like equation (equation A6 from Appendix Ward et al 2012) ###
        self.R = 0.05  # temperature depedence, {no units}
        self.T_ref = 293.15  # Reference Temperature at which gama_T=1, 20oC {Kelvin}
        self.gamma_T = np.exp(self.R * (self.T - self.T_ref))
        
        ### other plankton parameters (from Ward et al., 2012;2014) ###
        self.mp = 0.02  # plankton mortality {day^-1} 
        self.m=np.zeros([1,self.n_pft]) # mortality  
        self.lamdaprey = -1.0  # prey refure parameter {no units}
        self.thita_opt = 10.0  # optimum predator:prey ratio {no units}
        self.lamda = 0.7  # maximum assimilation efficiency
        self.sigma = 0.5 # standard deviation of zooplankton prey preference
        self.n_dt = 2.0 # number of timesteps: redundant currently
        self.B_init_value = 1e-4 # initial biomass concentrations
        self.K = 0.01  # Chemostat mixing rate {day^-1}
        
        self.pred_model='foodweb' # or foodchain
        
        ### planktonic foraminifera's parameters ###
        self.sigma_f = 0.5 # standard deviation of planktonic foraminifera's prey preference 
        self.gf = np.zeros([1,self.n_pft]) # grazing pressure on planktonic foraminifera
        
    #--------------------------------------------------------------------------
    #
    # function: initialise
    #
    # - calculates size-dependent parameters
    # - sets up model arrays
    #
    #--------------------------------------------------------------------------
    
    def initialise(self):
        
        self.pcmax_a=np.zeros([1,self.n_pft]) # Maximum photosynthetic rate
        self.pcmax_a[0,0:25]=[1.0,1.0,1.4,1.4,1.4,2.1,2.1,2.1,2.1,2.1,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8]
        
        ### Size Dependent Parameters ###
        ### phytoplankton ###
        self.muinf = np.zeros([1,self.n_pft])
        self.Qmin = np.zeros([1,self.n_pft])  # minimum nitrogen:carbon quoata for phyto {mmolN(mmolC)^-1} 
        self.Qmax = np.zeros([1,self.n_pft])  # maximum nitrogen:carbon quoata for phyto {mmolN(mmolC)^-1}
        self.deltaQ = np.zeros([1,self.n_pft])
        self.mumax = np.zeros([1,self.n_pft])
        self.kNO3 = np.zeros([1,self.n_pft])
        self.Vmax = np.zeros([1,self.n_pft])  # maximum uptake rate of phytoplankton{mmolN(mmolC)^-1}
        ### zooplankton ###
        self.kN = np.zeros([1,self.n_pft])  # Half- saturation concentration {mmolNm^-3}
        self.Gmax=np.zeros([1,self.n_pft])
        self.g=np.zeros([1,self.n_pft])
        self.kappa=np.zeros([1,self.n_pft])
        ### mortality ###
        self.mz=np.zeros([1,self.n_pft])
        self.m=np.zeros([1,self.n_pft])
        
        for n in range(0,self.n_pft):
            self.Vmax[0,n]=self.Vmax_a*self.V[0,n]**self.Vmax_b
            self.Qmax[0,n]=self.Qmax_a*self.V[0,n]**self.Qmax_b    
            self.Qmin[0,n]=self.Qmin_a*self.V[0,n]**self.Qmin_b  
            self.deltaQ[0,n]=self.Qmax[0,n]-self.Qmin[0,n]
            self.kNO3[0,n]=self.k_no3_a*self.V[0,n]**self.k_no3_b
            self.muinf[0,n]=self.pcmax_a[0,n]*self.V[0,n]**self.pcmax_b
            self.Gmax[0,n]=self.Gcmax_a*self.V[0,n]**self.Gcmax_b
            
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
                if self.pred_model=='foodchain':
                    self.mz[0,n]=self.mz_a*self.V[0,n]**self.mz_b
                    self.m[0,n]=self.mz[0,n] # size-dependent grazing
                elif self.pred_model=='foodweb':
                    self.mz[0,n] = self.mp 
                    self.m[0,n]= self.mz[0,n]# fixed grazing rate
                    #
            elif self.PFT_F[n]:
                self.g[0,n]=self.Gmax[0,n] *0.5 # * x value (ranges from 0 to 1 for energetic loss)
                self.gf[0,n] = 1.0 # grazing pressure on forams, ranges from 0 to 1 for different predation pressure on forams
                self.kappa[0,n]=self.K
                if self.pred_model=='foodchain':
                    self.mz[0,n]=(self.mz_a*self.V[0,n]**self.mz_b) 
                    self.m[0,n]= self.mz[0,n]  # size-dependent mortality
                elif self.pred_model=='foodweb':
                    self.mz[0,n] = self.mp * 0.81# * x value (ranges from 0 to 1 for protection from background mortality)
                    self.m[0,n]= self.mz[0,n] 
                    
                
        # Grazing matrix
        self.f=np.zeros([self.n_pft,self.n_pft]) # prey preference
        
        for jpred in range(0,self.n_pft): # loop over plankton
            if self.PFT_Z[jpred]: # if a zooplankton
                for jprey in range(0,self.n_pft): # loop over prey, i.e. all plankton
                    self.f[jprey,jpred] = np.exp(-(((np.log((self.D[0,jpred]/self.D[0,jprey])/ self.thita_opt)) ** 2) * ((2 * self.sigma ** 2) ** -1)))  # prey palatability #equation A21 from Appendix Ward et al 2012
                    
                    if self.PFT_Z[jprey] and self.pred_model=='foodchain':
                        self.f[jprey,jpred]=0.0
            if self.PFT_F[jpred]: # if forams
                for jprey in range(0,25): # loop over phytoplankton prey
                    self.f[jprey,jpred] = np.exp(-(((np.log((self.D[0,jpred]/self.D[0,jprey])/ self.thita_opt)) ** 2) * ((2 * self.sigma_f ** 2) ** -1)))  # prey palatability #equation A21 from Appendix Ward et al 2012
                    
                        
        self.graze_dt=np.zeros([self.n_pft,self.n_pft])
        #self.graze_dt=np.zeros([self.save_freq,self.n_pft])   
        
        # >>>>>>>>>>>>>>  initialise arrays
        self.dt=1.0/self.n_dt
        
        self.output_B=np.zeros([self.save_freq,self.n_pft]) # output array for plankton biomass through time
        self.output_N=np.zeros([self.save_freq,1]) # output array for nutrient through time
        
        self.N_init=np.ones([1,1])*1e-4 # nutrient array at t
        self.B_init=np.ones([1,self.n_pft])*self.B_init_value # biomass array at t
        
        self.output_t=np.zeros([self.save_freq,1])
        
        return
    
    #--------------------------------------------------------------------------
    #
    # function: run
    #
    # - calls initialise function to set model up
    # - calls dXdt function with solver
    # - returns output solution
    #
    #--------------------------------------------------------------------------
    
    def run(self):
        
        tstart=time.time()
        
        # initialise model 
        self.initialise()
        t=np.linspace(1.0,self.nday,self.save_freq)
        y0=np.append(self.N_init,self.B_init)
        
        ### call solver ###
        sol=odeint(self.dXdt,y0,t,full_output=True,rtol=1e-3,atol=1e-9,hmax=28.0,h0=0.01,hmin=0.001)
        
        sol_info=sol[1]
        sol=sol[0]
        
        self.run_status=sol_info['message']
        
        tend=time.time()
        self.run_time=tend-tstart # runtime in seconds
        
        # return solver output to output arrays
        self.output_t=t
        self.output_N[:,0]=sol[:,0]
        self.output_B[:,:]=sol[:,1:self.n_pft+1]

        ## weighted mean diameter
        ## calculate biomass proportion of each PFT
        biomass_prop = self.output_B/np.sum(self.output_B, axis=1)[:,None]
        ## calculate weighted mean diameter
        self.output_size = np.sum(biomass_prop*self.D, axis=1)

        
    #--------------------------------------------------------------------------
    #
    # function: dXdt
    #
    # - calculates dNdt and dBdt from model equations
    #
    #--------------------------------------------------------------------------
    
    def dXdt(self,X,t):
        
        N=np.zeros([1,1])
        B=np.zeros([1,self.n_pft])
        N[0,0]=X[0]
        B[0,:]=X[1:self.n_pft+1]
        Nuptake=np.zeros([1,self.n_pft])
        
        dBdt=np.zeros([1,self.n_pft])
        dNdt=np.zeros([1,1])
        graze=0.0
        
        
        for p in range(0,self.n_pft):
            
            
            if self.PFT_P[p]: # phyto
                
                Nuptake[0,p]=self.lambda_li* self.gamma_T * self.mumax[0,p]*N/(self.kN[0,p]+N)*B[0,p]
                

                
            elif self.PFT_Z[p]: # zoo
                
                F=np.sum(self.f[:,p]*B) # availability of prey
                
                if F>0.0:
                    PR= 1.0-np.exp(self.lamdaprey*F)  
                    
                    for jprey in range(0,self.n_pft): # loop over prey
                        
                        if self.PFT_P[jprey]:
                            pref=np.sum(self.f[self.PFT_P,p]*B[0,self.PFT_P]**2) / np.sum(self.f[:,p]*B**2)
                        elif self.PFT_Z[jprey]:
                            pref=np.sum(self.f[self.PFT_Z,p]*B[0,self.PFT_Z]**2) / np.sum(self.f[:,p]*B**2)
                        elif self.PFT_F[jprey]:
                            PR = 1.0 #no prey refuge
                            pref=np.sum(self.f[self.PFT_F,p]*B[0,self.PFT_F]**2) / np.sum(self.f[:,p]*B**2) 
                        if pref>1.0:
                            print('wrong value, pref>1.0')
                            
                             
                        graze=self.g[0,p]*self.gamma_T*(self.f[jprey,p]*B[0,jprey]/(F+self.kcprey)) * PR * pref * self.gf[0, jprey] #Holling Type II
                        
                                                                
                        dBdt[0,p]=dBdt[0,p]+(graze*B[0,p]*self.lamda)
                        
                        dBdt[0,jprey]=dBdt[0,jprey]-(graze*B[0,p]) 
                        
                        self.graze_dt[jprey,p]=graze
                        
            elif self.PFT_F[p]: # forams
                F=np.sum(self.f[:,p]*B) # availability of prey
                if F>0.0:
                    PR= 1.0-np.exp(self.lamdaprey*F)
                    
                    for jprey in range(0,25): # loop over phytoplankton prey
                        
                        graze = self.g[0,p]*self.gamma_T*(self.f[jprey,p]*B[0,jprey]/(F+self.kcprey)) * PR 
                        
                        dBdt[0,p]=dBdt[0,p]+(graze*B[0,p]*self.lamda) 
                        
                        dBdt[0,jprey]=dBdt[0,jprey]-(graze*B[0,p]) 
                        
                        self.graze_dt[jprey,p]=graze
                        
                        

        self.Nuptake=Nuptake      
        dBdt=dBdt+Nuptake-(B*self.m)-(self.kappa*B) 
        dNdt=self.K*(self.No-N) - np.sum(Nuptake)
        ##non negative values 
        if dBdt[0,p] < 1e-99:
            dBdt[0,p] = 1e-99
            self.graze_dt[0,p] = 0.0
        if dNdt[0] < 1e-99:
            dNdt[0] = 1e-99
            Nuptake = 0.0
        return np.append(dNdt,dBdt)
    
