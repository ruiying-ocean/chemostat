"""Ecological rate functions, independent of configuration and the ODE solver.

Uptake and loss functions return rates. Grazing functions add predator gains and
prey losses to the supplied biomass derivative, ``dBdt``, in place. Biomass and
nutrient inputs must be nonnegative; the model floors solver overshoots to zero.
"""

import numpy as np


def nutrient_uptake(N, B, gamma_T, *, light, mumax, kN, can_uptake):
    """Monod uptake for phytoplankton and symbiont-bearing forams."""
    out = np.zeros_like(B, dtype=float)
    u = can_uptake
    out[u] = light[u] * gamma_T[u] * mumax[u] * N / (kN[u] + N) * B[u]
    return out


def zooplankton_grazing(
    i, B, gamma_T, dBdt, *, preference, Gmax, refuge_strength,
    half_saturation, assimilation, prey_groups,
):
    """Size-selective grazing with switching among phyto, zoo, and foram prey.

    ``prey_groups`` contains the three boolean masks in that order. Forams are
    not protected by the low-prey refuge term.
    """
    f_i = preference
    F = np.sum(f_i * B)
    if F <= 0.0:
        return
    PR = 1.0 - np.exp(refuge_strength * F)
    # Summing category totals keeps weights <= 1 despite rounding.
    cat_sum = [np.sum(f_i[c] * B[c] ** 2) for c in prey_groups]
    denom = sum(cat_sum)
    if denom <= 0.0:  # prey biomass so small that B**2 underflows
        return
    pref = np.zeros_like(B, dtype=float)
    for c, s in zip(prey_groups, cat_sum):
        pref[c] = s / denom

    pr_eff = np.where(prey_groups[2], 1.0, PR)
    graze = Gmax * gamma_T * f_i * B / (F + half_saturation) * pr_eff * pref
    dBdt[i] += np.sum(graze) * B[i] * assimilation
    dBdt -= graze * B[i]


def foram_grazing(
    i, B, gamma_T, dBdt, *, preference, Gmax, refuge_strength,
    half_saturation, assimilation, switching_exponent,
):
    """Holling type II grazing with morphotype-specific prey switching.

    The switching weights are (preference * biomass)**ns, as in ECOGEM.
    ns=1 gives the legacy response; ns=5 concentrates feeding on preferred prey.
    """
    f_i = preference
    F_total = np.sum(f_i * B)
    if F_total <= 0.0:
        return
    PR = 1.0 - np.exp(refuge_strength * F_total)

    food_ns = (f_i * B) ** switching_exponent
    W = np.sum(food_ns)
    if W <= 0.0:
        return

    total_rate = Gmax * gamma_T * F_total / (F_total + half_saturation) * PR
    graze = total_rate * food_ns / W
    dBdt[i] += np.sum(graze) * B[i] * assimilation
    dBdt -= graze * B[i]


def mortality(B, rates):
    """Biomass lost to mortality, per PFT."""
    return B * rates


def dilution(B, rate):
    """Biomass washed out of the chemostat, per PFT."""
    return rate * B


def respiration(B, gamma_T, rates):
    """Temperature-dependent biomass loss, per PFT."""
    return B * rates * gamma_T


def nutrient_supply(N, source_N, dilution_rate):
    """Net nutrient exchange with the source water."""
    return dilution_rate * (source_N - N)


DEFAULT_PROCESSES = {
    "uptake": nutrient_uptake,
    "zooplankton_grazing": zooplankton_grazing,
    "foram_grazing": foram_grazing,
    "mortality": mortality,
    "dilution": dilution,
    "respiration": respiration,
    "nutrient_supply": nutrient_supply,
}
