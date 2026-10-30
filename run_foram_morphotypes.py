"""Run the chemostat with the four foram morphotypes and plot community structure.

Foram sizes and morphotypes come from model_config.toml (esd_foram / type_foram),
one representative size per morphotype as in FORAMECOGEM 2P3D2E4Z4F.eco.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from chemostat import Chemostat


MORPHO_COLORS = {
    'bn': '#1f77b4',
    'bs': '#2ca02c',
    'sn': '#ff7f0e',
    'ss': '#d62728',
}
MORPHO_LABELS = {
    'bn': 'bn (barren, non-spinose)',
    'bs': 'bs (barren, spinose)',
    'sn': 'sn (symbiont, non-spinose)',
    'ss': 'ss (symbiont, spinose)',
}


def build_model(temp: float, nutrient: float) -> Chemostat:
    m = Chemostat(temp, nutrient)
    m.load_ecoconfig('model_config.toml')
    return m


def plot_community(m: Chemostat, save_path: str | None = None) -> None:
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), tight_layout=True)

    F_idx = np.flatnonzero(m.PFT_F_bool)
    foram_types = m.foram_type[F_idx]
    B_F = m.output_B[:, F_idx]
    diam_F = m.diameter[F_idx]
    t = m.output_t

    # 1) total biomass by group (P, Z, F)
    ax = axs[0, 0]
    ax.plot(t, m.output_B[:, m.PFT_P_bool].sum(axis=1), label='Phytoplankton', color='tab:green')
    ax.plot(t, m.output_B[:, m.PFT_Z_bool].sum(axis=1), label='Zooplankton', color='tab:blue')
    ax.plot(t, m.output_B[:, m.PFT_F_bool].sum(axis=1), label='Foraminifera', color='tab:red')
    ax.set_title('Community biomass')
    ax.set_ylabel('Biomass (mmol N m$^{-3}$)')
    ax.set_xlabel('Time (days)')
    ax.legend()

    # 2) foram biomass by morphotype
    ax = axs[0, 1]
    for mtype in ('bn', 'bs', 'sn', 'ss'):
        mask = foram_types == mtype
        if not np.any(mask):
            continue
        ax.plot(t, B_F[:, mask].sum(axis=1),
                color=MORPHO_COLORS[mtype], label=MORPHO_LABELS[mtype], linewidth=2)
    ax.set_title('Foram biomass by morphotype')
    ax.set_ylabel('Biomass (mmol N m$^{-3}$)')
    ax.set_xlabel('Time (days)')
    ax.legend(fontsize=8)

    # 3) stacked relative composition of the foram community
    ax = axs[1, 0]
    total_F = B_F.sum(axis=1)
    total_F = np.where(total_F > 0, total_F, np.nan)
    stack = []
    labels = []
    colors = []
    for mtype in ('bn', 'bs', 'sn', 'ss'):
        mask = foram_types == mtype
        if not np.any(mask):
            continue
        stack.append(B_F[:, mask].sum(axis=1) / total_F)
        labels.append(MORPHO_LABELS[mtype])
        colors.append(MORPHO_COLORS[mtype])
    ax.stackplot(t, *stack, labels=labels, colors=colors, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title('Foram community composition (relative)')
    ax.set_ylabel('Fraction of foram biomass')
    ax.set_xlabel('Time (days)')
    ax.legend(loc='upper right', fontsize=8)

    # 4) final biomass vs size, coloured by morphotype
    ax = axs[1, 1]
    final = B_F[-1, :]
    for mtype in ('bn', 'bs', 'sn', 'ss'):
        mask = foram_types == mtype
        if not np.any(mask):
            continue
        ax.scatter(diam_F[mask], final[mask],
                   color=MORPHO_COLORS[mtype], s=60, label=MORPHO_LABELS[mtype],
                   edgecolor='k', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Final biomass vs size (t={int(t[-1])} d)')
    ax.set_xlabel('ESD (μm)')
    ax.set_ylabel('Biomass (mmol N m$^{-3}$)')
    ax.legend(fontsize=8)

    fig.suptitle('Foram community structure with symbiotic / spinose morphotypes',
                 fontsize=12, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'saved figure: {save_path}')
    else:
        plt.show()


def main() -> None:
    m = build_model(temp=20.0, nutrient=2.5)
    print('morphotype assignment:', list(zip(m.type_foram, m.esd_foram)))
    m.run()
    plot_community(m, save_path='docs/foram_morphotypes.png')


if __name__ == '__main__':
    main()
