"""Sweep temperature x source-nitrate and compare foram morphotype response.

For each (T, N) cell we run the chemostat to steady state and record the foram-
community composition averaged over the last 10% of the run (to smooth bloom
transients). Output: a grid of heatmaps, one per morphotype, plus total foram
biomass and the dominant morphotype map.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from chemostat import Chemostat

MORPHOTYPES = ('bn', 'bs', 'sn', 'ss')
MORPHO_COLORS = {'bn': '#1f77b4', 'bs': '#2ca02c', 'sn': '#ff7f0e', 'ss': '#d62728'}
# total foram biomass below 1% of init_B counts as extinct: fractions there are solver noise
EXTINCT = 1e-6

MORPHO_LABELS = {
    'bn': 'bn (barren, non-spinose)',
    'bs': 'bs (barren, spinose)',
    'sn': 'sn (symbiont, non-spinose)',
    'ss': 'ss (symbiont, spinose)',
}

def run_one(temp: float, nutrient: float, config: str) -> dict:
    m = Chemostat(temp, nutrient)
    m.load_ecoconfig(config)
    m.run()  # a solver failure should surface, not be retried with a non-stiff method

    F_idx = np.flatnonzero(m.PFT_F_bool)
    foram_types = m.foram_type[F_idx]
    # average over the final 10% of the simulation
    tail = max(1, int(0.1 * m.output_B.shape[0]))
    B_F_tail = m.output_B[-tail:, F_idx]
    B_F_mean = np.maximum(B_F_tail.mean(axis=0), 0.0)  # extinct types sit at ~ -1e-30 under the solver's floor
    total = B_F_mean.sum()

    result = {'total_F': float(total)}
    for mt in MORPHOTYPES:
        mask = foram_types == mt
        result[f'B_{mt}'] = float(B_F_mean[mask].sum())
        result[f'frac_{mt}'] = float(B_F_mean[mask].sum() / total) if total > EXTINCT else np.nan
    return result


def main() -> None:
    # usage: run_foram_TN_sweep.py [config.toml]; model_config_X.toml saves foram_TN_sweep_X.png
    config = sys.argv[1] if len(sys.argv) > 1 else 'model_config.toml'
    suffix = Path(config).stem.removeprefix('model_config').lstrip('_')
    save_path = f"docs/foram_TN_sweep{'_' + suffix if suffix else ''}.png"
    temps = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    nutrients = np.array([0.25, 0.5, 1.0, 2.5, 5.0, 10.0])

    nT, nN = len(temps), len(nutrients)
    totals = np.zeros((nT, nN))
    fracs = {mt: np.zeros((nT, nN)) for mt in MORPHOTYPES}

    for i, T in enumerate(temps):
        for j, N in enumerate(nutrients):
            print(f'  T={T:>4.1f} °C, source_N={N:>5.2f} ...', end=' ')
            r = run_one(T, N, config)
            totals[i, j] = r['total_F']
            for mt in MORPHOTYPES:
                fracs[mt][i, j] = r[f'frac_{mt}']
            print(f"total={r['total_F']:.3e}  bn={r['frac_bn']:.2f}  bs={r['frac_bs']:.2f}  "
                  f"sn={r['frac_sn']:.2f}  ss={r['frac_ss']:.2f}")

    plot_results(temps, nutrients, totals, fracs, save_path=save_path, config=config)


def plot_results(temps, nutrients, totals, fracs, save_path, config):
    plt.style.use('default')
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), tight_layout=True)

    # the N grid is uneven, so draw cells by index and label ticks with the real values
    def label_axes(ax):
        ax.set_xticks(range(len(nutrients)), [f'{n:g}' for n in nutrients])
        ax.set_yticks(range(len(temps)), [f'{t:g}' for t in temps])
        ax.set_xlabel('source N (mmol N m$^{-3}$)')
        ax.set_ylabel('Temperature (°C)')

    def imshow(ax, data, title, cmap='viridis', vmin=None, vmax=None, fmt='{:.2f}'):
        im = ax.imshow(data, origin='lower', aspect='auto',
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        label_axes(ax)
        ax.set_title(title)
        nT, nN = data.shape
        for i in range(nT):
            for j in range(nN):
                if np.isnan(data[i, j]):
                    continue
                ax.text(j, i, fmt.format(data[i, j]),
                        ha='center', va='center', fontsize=7,
                        color='white' if cmap == 'viridis' and data[i, j] < (vmax or data.max()) * 0.6 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # one heatmap per morphotype (fractional)
    for k, mt in enumerate(MORPHOTYPES):
        ax = axs.flat[k]
        imshow(ax, fracs[mt], f'{MORPHO_LABELS[mt]}\n(fraction of foram biomass)',
               cmap='viridis', vmin=0.0, vmax=1.0)

    # total foram biomass
    ax = axs[1, 1]
    imshow(ax, totals, 'Total foram biomass\n(mmol N m$^{-3}$, mean over final 10%)',
           cmap='magma', fmt='{:.2e}')

    # dominant morphotype map
    ax = axs[1, 2]
    stacked = np.stack([fracs[mt] for mt in MORPHOTYPES], axis=-1)
    dominant = np.argmax(np.nan_to_num(stacked), axis=-1).astype(float)
    dominant[totals < EXTINCT] = np.nan  # no winner where forams are extinct
    cmap = ListedColormap([MORPHO_COLORS[mt] for mt in MORPHOTYPES])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.imshow(dominant, origin='lower', aspect='auto',
                   cmap=cmap, norm=norm, interpolation='nearest')
    label_axes(ax)
    ax.set_title('Dominant morphotype')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(list(MORPHOTYPES))

    fig.suptitle(f'Foram morphotype response to temperature and nutrient supply ({config})',
                 fontsize=13, fontweight='bold')
    fig.savefig(save_path, dpi=150)
    print(f'saved figure: {save_path}')


if __name__ == '__main__':
    main()
