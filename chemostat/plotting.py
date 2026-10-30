"""Figures for completed chemostat simulations."""

import matplotlib.pyplot as plt
import numpy as np


def plot(model, sum_PFT=True, save_path=None):
    ### plot biomass, nutrient, size distribution
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2,2, figsize=(8, 8), sharex=True, sharey=False, tight_layout=True)

    ## Set shared x axis
    fig.supxlabel('Time (days)')


    if sum_PFT:
        axs[0,0].plot(model.output_t, np.sum(model.output_B[:,model.PFT_P_bool], axis=1), label='Phytoplankton')
        axs[0,0].plot(model.output_t, np.sum(model.output_B[:,model.PFT_Z_bool], axis=1), label='Zooplankton')
        axs[0,0].plot(model.output_t, np.sum(model.output_B[:,model.PFT_F_bool], axis=1), label='Foraminifera')
        axs[0,0].legend()
    else:
        axs[0,0].plot(model.output_t, model.output_B[:,model.PFT_P_bool])
        axs[0,0].plot(model.output_t, model.output_B[:,model.PFT_Z_bool])
        axs[0,0].plot(model.output_t, model.output_B[:,model.PFT_F_bool])

    axs[0,0].set_title('Biomass')
    axs[0,0].set_ylabel('Biomass (mmol N m$^{-3}$)')

    axs[0,1].plot(model.output_t, model.output_N, linewidth=1)
    axs[0,1].set_title('Nutrient')
    axs[0,1].set_ylabel('Nutrient (mmol N m$^{-3}$)')

    axs[1,0].plot(model.output_t, model.output_size, linewidth=1)
    axs[1,0].set_title('Community Size Mean')
    axs[1,0].set_ylabel('Size (μm)')

    axs[1,1].plot(model.output_t, model.output_pft_richness, linewidth=1)
    axs[1,1].set_title('PFT Richness')
    axs[1,1].set_ylabel('Number of PFTs')

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return axs
