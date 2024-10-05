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

    def diff_eqn(self, y, t):
        """differential equation fed to odeint solver
        y: state variable (including both N and B)
        t: time
        """

        ## state variable
        N = np.zeros(1)
        B = np.zeros(self.n_pft)
        F = np.zeros(self.n_pft)
        
        ## initial condition
        N = y[0]
        B = y[1:self.n_pft + 1]

        Nuptake = np.zeros(self.n_pft)

        ## array for the rate of change
        dBdt = np.zeros(self.n_pft)
        dNdt = np.zeros(1)

        self.gamma_T = np.exp(self.R * (self.T - self.T_ref))

        Nuptake = self.gamma_l * self.gamma_T * self.mumax * N / (self.kN + N) * B
        dNdt = dNdt + self.K * (self.source_N - N) - np.sum(Nuptake)

        F = np.dot(self.f.T, B)
        graze_rate = self.Gmax * self.gamma_T * F / (F + self.knprey)
        
        ## grazing term
        for i in range(self.n_pft):
            for j in range(self.n_pft):
                ## i = predator, j = prey
                F[i] = np.sum(self.f[j,i] * B[j])
                # if F[i] == F_vec[i]:
                #     print('F_vec[i] is correct')

                graze_rate[i] = self.Gmax[i] * self.gamma_T * F[i] / (F[i] + self.knprey)
                
                gross_graz = graze_rate[i] * B[i]
                
                if gross_graz > B[j]: gross_graz = B[j]
                    
                dBdt[i] = dBdt[i] + gross_graz * self.lamda
                dBdt[j] = dBdt[j] - gross_graz
                
        
        dBdt = dBdt+ Nuptake - (B * self.m) - (self.K * B) - (B * self.resp * self.gamma_T)

        return np.append(dNdt, dBdt)


    def plot(self):
        ### plot biomass, nutrient, size distribution
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=False)

        ## Set shared x axis
        fig.supxlabel('Time (days)')

        axs[0].plot(self.output_t, self.output_B[:,0:self.n_phytoplankton], 'green')
        axs[0].plot(self.output_t, self.output_B[:,self.n_phytoplankton:self.n_phytoplankton+self.n_zooplankton],'red')
        axs[0].plot(self.output_t, self.output_B[:,self.n_phytoplankton+self.n_zooplankton:self.n_pft],'black')
        axs[0].set_title('Biomass')
        axs[0].set_ylabel('Biomass (mmol N m$^{-3}$)')

        axs[1].plot(self.output_t, self.output_N, color='black', linewidth=2)
        axs[1].set_title('Nutrient')
        axs[1].set_ylabel('Nutrient (mmol N m$^{-3}$)')

        axs[2].plot(self.output_t, self.output_size, color='black', linewidth=2)
        axs[2].set_title('Community Size Mean')
        axs[2].set_ylabel('Size (μm)')

        
        

        

        plt.show()
        return axs
