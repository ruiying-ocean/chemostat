# -*- coding: utf-8 -*-
"""
Model code for the study "A trait-based modelling approach to planktonic foraminifera ecology"

code authors: Maria Grigoratou, Jamie Wilson and Ben Ward
contact author: Maria Grigoratou, maria.grigoratou@bristol.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import Plankton_Size_Chemostat

## Model input
## runtime, save_freq, temperature, nutrient concentration, diameters, plankton_type

##diameters 
diameter = np.array([[6.20400000e-01,   7.81655019e-01,   9.84823613e-01,   1.24080000e+00,
    1.56331004e+00,   1.96964723e+00,   2.48160000e+00,   3.12662008e+00,
    3.93929445e+00,   4.96320000e+00,   6.25324015e+00,   7.87858890e+00,
    9.92640000e+00,   1.25064803e+01,   1.57571778e+01,   1.98528000e+01,
    2.50129606e+01,   3.15143556e+01,   3.97056000e+01,   5.00259212e+01,
    6.30287112e+01,   7.94112000e+01,   1.00051842e+02,   1.26057422e+02,
    1.58822400e+02,  
    6.25324015e+00,   7.87858890e+00,   9.92640000e+00,
    1.25064803e+01,   1.57571778e+01,   1.98528000e+01,   2.50129606e+01,
    3.15143556e+01,   3.97056000e+01,   5.00259212e+01,   6.30287112e+01,
    7.94112000e+01,   1.00051842e+02,   1.26057422e+02,   1.58822400e+02,
    2.00103685e+02,   2.52114845e+02,   3.17644800e+02,   4.00207370e+02,
    5.04229690e+02,   6.35289600e+02,   8.00414740e+02,   1.00845938e+03,
    1.27057920e+03,   1.60082948e+03, 
    
    6.25324015e+00,   7.87858890e+00,   9.92640000e+00,
    1.25064803e+01,   1.57571778e+01,   1.98528000e+01,   2.50129606e+01,
    3.15143556e+01,   3.97056000e+01,   5.00259212e+01,   6.30287112e+01,
    7.94112000e+01,   1.00051842e+02,   1.26057422e+02,   1.58822400e+02,
    2.00103685e+02,   2.52114845e+02,   3.17644800e+02,   4.00207370e+02,
    5.04229690e+02,   6.35289600e+02,   8.00414740e+02,   1.00845938e+03,
    1.27057920e+03,   1.60082948e+03]], ndmin =2) # the size for foram prolocular equals to 1.98528000e+01 and the size for foram adult  equals to1.58822400e+02

## set up plankton types
n_pft=diameter.size
plankton_type=np.zeros([1,n_pft])
plankton_type[0,0:25]=1
plankton_type[0,25:50]=2
plankton_type[0,50:51]=3

## Nutrient concentration gradient
nutrient_init=2.5
sst= 25

## create model instance
model=Plankton_Size_Chemostat(1000,100, sst, nutrient_init, diameter, plankton_type)
model.pred_model='foodweb'
model.thita_opt=diameter[0,25]/diameter[0,0]
model.run()

## Biomass state variable
t= model.output_t
biomass = model.output_B
nutrient = model.output_N
size = model.output_size

### plot biomass, nutrient, size distribution
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=False)

## Set shared x axis
fig.supxlabel('Time (days)')

axs[0].plot(t, biomass[:,0:25],'r')
axs[0].plot(t, biomass[:,25:50],'b')
axs[0].plot(t, biomass[:,51:75],'black')
axs[0].set_title('Biomass')
axs[0].set_ylabel('Biomass (mmol N m$^{-3}$)')

axs[1].plot(t, nutrient, color='black', linewidth=2)
axs[1].set_title('Nutrient')
axs[1].set_ylabel('Nutrient (mmol N m$^{-3}$)')

axs[2].plot(t, size, color='black', linewidth=2)
axs[2].set_title('Community Size Mean')
axs[2].set_ylabel('Size (μm)')

plt.show()

