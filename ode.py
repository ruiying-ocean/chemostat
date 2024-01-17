import numpy as np
import time
from scipy.integrate import odeint


class ODE:
    def run(self):
        ## time
        t = np.linspace(1.0, self.nday, self.save_freq)
        ## init condition
        y0 = np.append(self.N0, self.B0)

        ### call solver ###
        ## dydt is the function describing the system, in the form of f(y,t);
        ## y could be multi-dimension, but t must be 1-dimension
        sol = odeint(self.dydt, y0, t, full_output=False, rtol=1e-3, atol=1e-9, hmax=28.0, h0=0.01, hmin=0.001)

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

    def dydt(self, y, t):
        "describe the dynamic of the system"

        N = np.zeros([1, 1])
        B = np.zeros([1, self.n_pft])
        
        ## initial condition
        N[0, 0] = y[0]
        B[0, :] = y[1:self.n_pft + 1]
        Nuptake = np.zeros([1, self.n_pft])

        dBdt = np.zeros([1, self.n_pft])
        dNdt = np.zeros([1, 1])
        graze = 0.0

        self.gamma_T = np.exp(self.R * (self.T - self.T_ref))

        for p in range(0, self.n_pft):
            if self.PFT_P[p]:
                Nuptake[0, p] = self.gamma_l * self.gamma_T * self.mumax[0, p] * N / (self.kN[0, p] + N) * B[0, p]
            elif self.PFT_Z[p]:
                F = np.sum(self.f[:, p] * B)  # availability of prey
                if F > 0.0:
                    PR = 1.0 - np.exp(self.lamdaprey * F)

                    for jprey in range(0, self.n_pft):  # loop over prey

                        if self.PFT_P[jprey]:
                            pref = np.sum(self.f[self.PFT_P, p] * B[0, self.PFT_P] ** 2) / np.sum(self.f[:, p] * B ** 2)
                        elif self.PFT_Z[jprey]:
                            pref = np.sum(self.f[self.PFT_Z, p] * B[0, self.PFT_Z] ** 2) / np.sum(self.f[:, p] * B ** 2)
                        elif self.PFT_F[jprey]:
                            pref = np.sum(self.f[self.PFT_F, p] * B[0, self.PFT_F] ** 2) / np.sum(self.f[:, p] * B ** 2)
                        if pref > 1.0:
                            print('wrong value, pref>1.0')

                        graze = self.g[0, p] * self.gamma_T * (
                                    self.f[jprey, p] * B[0, jprey] / (F + self.kcprey)) * PR * pref * self.gf[
                                    0, jprey]  # Holling Type II

                        dBdt[0, p] = dBdt[0, p] + (graze * B[0, p] * self.lamda)

                        dBdt[0, jprey] = dBdt[0, jprey] - (graze * B[0, p])

                        self.graze_dt[jprey, p] = graze

            elif self.PFT_F[jprey]:
                F = np.sum(self.f[:, p] * B)  # availability of prey
                PR = 1.0 #no prey refuge
                if F > 0.0:
                    PR = 1.0 - np.exp(self.lamdaprey * F)

                    for jprey in range(0, self.n_phytoplankton):  # loop over phytoplankton prey

                        graze = self.g[0, p] * self.gamma_T * (self.f[jprey, p] * B[0, jprey] / (F + self.kcprey)) * PR

                        dBdt[0, p] = dBdt[0, p] + (graze * B[0, p] * self.lamda)

                        dBdt[0, jprey] = dBdt[0, jprey] - (graze * B[0, p])

                        self.graze_dt[jprey, p] = graze

        dBdt = dBdt + Nuptake - (B * self.m) - (self.kappa * B)
        dNdt = self.K * (self.source_N - N) - np.sum(Nuptake)
        ##non negative values
        if dBdt[0, p] < 1e-99:
            dBdt[0, p] = 1e-99
            self.graze_dt[0, p] = 0.0
        if dNdt[0] < 1e-99:
            dNdt[0] = 1e-99
            Nuptake = 0.0
        return np.append(dNdt, dBdt)
