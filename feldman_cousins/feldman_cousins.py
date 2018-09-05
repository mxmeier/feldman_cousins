#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, print_function

import warnings
from collections import Iterable
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np
import scipy.stats as scs
from matplotlib import pyplot as plt

from _utility import fix_monotonicity


def feldman_cousins_acceptance_interval_from_pdf(pdf, n_b=0, alpha=0.9):
    '''
    Acceptance intervals for the Feldman-Cousins approach given a
    non-analytical pdf.

    The Likelihood ranking suggested by Feldman and Cousins is performed
    empirical here. This calculation fails when mu_best is not sampled for the
    given PDF. In the case of a simple Poisson PDF with background this starts
    to fail as soon as n_obs - n_b is greater than the scanned range in mu.

    For more information see `Feldman-Cousins`_.

    Parameters
    ----------
    pdf: array-like, shape (len(mus), len(n_obs))
        PDF values calculated on a (mu, n_obs) grid.
        n_obs is assumed to be integer values and to start from 0.
    n_b: float, default: 0
        The value of n_b used to calculate the corresponding pdf can be passed
        to the function and will just be returned, can be used to safely
        identify processes after parallel processing.
    alpha: float
        The desired confidence level of the constructed confidence belt.

    Returns
    -------
    lower_limit: array-like, shape(len(mus))
        The lower limits on `n_obs` for each `mu`.
    upper_limit: array-like, shape(len(mus))
        The upper limits on `n_obs` for each `mu`.
    n_b: float
        Returns the n_b value given to the function without any modification.

    .. _Feldman-Cousins:
        https://arxiv.org/abs/physics/9711021
    '''
    pdf = np.array(pdf)

    n_obs = np.arange(pdf.shape[1])

    lower_limit_n = np.zeros(pdf.shape[0]) * np.nan
    upper_limit_n = np.zeros(pdf.shape[0]) * np.nan

    # Rank all the values for each n_obs
    L_best = np.amax(pdf, axis=0)
    L_best[L_best == 0] = 1

    pdf_rescaled = pdf / L_best

    for mu_idx in range(pdf.shape[0]):
        i = 0

        ranking = pdf_rescaled[mu_idx]
        indexes = np.argsort(ranking)[::-1]

        lower_limit_n[mu_idx] = n_obs[indexes[i]]
        upper_limit_n[mu_idx] = n_obs[indexes[i]]

        prob = pdf[mu_idx, indexes[i]]

        while prob <= alpha and i < len(n_obs) - 1:
            next_n = indexes[i+1]
            if next_n >= upper_limit_n[mu_idx]:
                upper_limit_n[mu_idx] = next_n
            if next_n <= lower_limit_n[mu_idx]:
                lower_limit_n[mu_idx] = next_n

            prob += pdf[mu_idx, next_n]

            i += 1

    upper_limit_n[upper_limit_n == n_obs[-1]] = np.nan

    # Add one n_bin_width to the upper limit to rather overcover the
    # interval than undercover
    n_bin_width = n_obs[1] - n_obs[0]

    lower_limit_n, upper_limit_n = fix_monotonicity(
        lower_limit_n, upper_limit_n)

    return lower_limit_n, upper_limit_n + n_bin_width, n_b


def feldman_cousins_intervals_from_acceptance_intervals(
        n_obs, n_b, mus,
        lower_limit_n,
        upper_limit_n,
        fix_discrete_n_pathology=True):
    '''
    Calculates confidence belts with the Feldman-Cousins approach
    from arbitrary acceptance intervals.

    For more information see `Feldman-Cousins`_.

    Parameters
    ----------
    n_obs: array-like
        Range of `n_obs` to scan while constructing the limits on
        `n_obs` for each `mu`.
    n_b: array-like
        Number of background events. Parameter for the poissonian
        distribution with background.
    mus: array-like
        Grid of `mu` values to contruct the limits on `n_obs` on.
        As `n_b` gets bigger the grid needs to get finer to actually
        populte every important `n_obs` value with an upper limit.
    lower_limit_n: array-like, shape(len(mus)) or (len(mus), len(n_b))
        Lower limit on n for each mu.
    upper_limit_n: array-like, shape(len(mus)) or (len(mus), len(n_b))
        Upper limit on n for each mu.
    fix_discrete_n_pathology: bool, optional
        If True, calculate the confidence belts for surrounding n_b to
        correct for a pathology arising from the discreteness of n_obs
        in the poissonian distribution, which causes some upper limits
        to rise for rising n_b.

    Returns
    -------
    lower_limit: array-like, shape(len(n_obs), len(n_b))
        The lower limits on `mu` for each `n_obs`.
    upper_limit: array-like, shape(len(n_obs), len(n_b))
        The upper limitson `mu` for each `n_obs`.

    .. _Feldman-Cousins:
        https://arxiv.org/abs/physics/9711021
    '''

    def fix_pathology(arr):
        for j in range(1, len(arr)):
            if arr[-(j + 1)] < arr[-j]:
                arr[-(j + 1)] = arr[-j]
        return arr

    if not isinstance(n_obs, Iterable):
        try:
            n_obs = np.arange(n_obs)
        except:
            raise ValueError('n_obs either has to be an Iterable ' +
                             'or an int, which is then fed into ' +
                             'np.arange.')

    if not isinstance(n_b, Iterable):
        n_b = np.array([n_b])

    lower_limit_n = np.atleast_2d(lower_limit_n)
    upper_limit_n = np.atleast_2d(upper_limit_n)

    if len(mus) != lower_limit_n.shape[0]:
        lower_limit_n = lower_limit_n.T
    if len(mus) != upper_limit_n.shape[0]:
        upper_limit_n = upper_limit_n.T

    # Translate upper and lower limit in measured n into
    # an interval in signal mean mu
    upper_limit_mu = np.zeros((len(n_obs), len(n_b)), dtype=float) - 1.
    lower_limit_mu = np.zeros((len(n_obs), len(n_b)), dtype=float) - 1.

    for n in range(len(n_obs)):
        for b in range(len(n_b)):
            upper_idx = upper_limit_n[:, b] == (n+1)
            lower_idx = lower_limit_n[:, b] == n

            if np.sum(lower_idx) == 0:
                warnings.warn(('The given `mus`-array is probably to coarse.' +
                               ' No upper limit found for `n_obs` = {}. ' +
                               'Setting the upper limit to inf.').format(n))
                upper_limit_mu[n, b] = np.inf
            else:
                upper_limit_mu[n, b] = mus[lower_idx][-1]
            if np.sum(upper_idx) == 0:
                lower_limit_mu[n, b] = 0
            else:
                lower_limit_mu[n, b] = mus[upper_idx][0]

        if fix_discrete_n_pathology:
            lower_limit_mu[n] = fix_pathology(lower_limit_mu[n])
            upper_limit_mu[n] = fix_pathology(upper_limit_mu[n])

    return lower_limit_mu, upper_limit_mu


def poissonian_feldman_cousins_acceptance_interval(n_obs, n_b, mus, alpha=0.9):
    '''
    Acceptance intervals for a poisson process with background for the Feldman-
    Cousins approach.
    For more information see `Feldman-Cousins`_.

    Parameters
    ----------
    n_obs: array-like
        Range of `n_obs` to scan while constructing the limits on
        `n_obs` for each `mu`.
    n_b: array-like
        Number of background events. Parameter of the poissonian
        distribution with background.
    mus: array-like
        Grid of `mu` values to construct the limits on `n_obs` on.
        As `n_b` gets bigger the grid needs to get finer to actually
        populate every important `n_obs` value with an upper limit.
    alpha: float
        The desired confidence level of the constructed confidence belt.

    Returns
    -------
    lower_limit: array-like, shape(len(mus), len(n_b))
        The lower limits on `n_obs` for each `mu`.
    upper_limit: array-like, shape(len(mus), len(n_b))
        The upper limits on `n_obs` for each `mu`.

    .. _Feldman-Cousins:
        https://arxiv.org/abs/physics/9711021
    '''
    def poisson_with_bg_rank(n_obs, n_b, mu=0):
        R = np.zeros_like(n_obs)
        L = scs.poisson.pmf(n_obs, mu + n_b)
        mu_best = np.maximum(0, n_obs - n_b)
        L_best = scs.poisson.pmf(n_obs, mu_best + n_b)
        R = L / L_best
        return R

    def poisson_with_bg_cdf(lower_limit, upper_limit, mu, n_b):
        x = np.arange(lower_limit, upper_limit + 1, 1)
        prob = np.sum(scs.poisson.pmf(x, mu + n_b))
        return prob

    if not isinstance(n_b, Iterable):
        n_b = [n_b]

    lower_limit_n = np.zeros((len(mus), len(n_b))) * np.nan
    upper_limit_n = np.zeros((len(mus), len(n_b))) * np.nan

    for mu_idx, mu in enumerate(mus):
        for nb_idx, nb in enumerate(n_b):
            i = 0

            ranking = poisson_with_bg_rank(n_obs, nb, mu)
            indexes = np.argsort(ranking)[::-1]

            lower_limit_n[mu_idx, nb_idx] = n_obs[indexes[i]]
            upper_limit_n[mu_idx, nb_idx] = n_obs[indexes[i]]

            prob = poisson_with_bg_cdf(lower_limit_n[mu_idx, nb_idx],
                                       upper_limit_n[mu_idx, nb_idx],
                                       mu, nb)

            while prob <= alpha and i < len(n_obs)-1:
                next_n = n_obs[indexes[i+1]]
                if next_n >= upper_limit_n[mu_idx, nb_idx]:
                    upper_limit_n[mu_idx, nb_idx] = next_n
                if next_n <= lower_limit_n[mu_idx, nb_idx]:
                    lower_limit_n[mu_idx, nb_idx] = next_n

                prob = poisson_with_bg_cdf(lower_limit_n[mu_idx, nb_idx],
                                           upper_limit_n[mu_idx, nb_idx],
                                           mu, nb)
                i += 1

    # Add one n_bin_width to the upper limit to rather overcover the
    # interval than undercover
    n_bin_width = n_obs[1] - n_obs[0]

    for nb_idx in range(len(n_b)):
        lower_limit_n[:, nb_idx], upper_limit_n[:, nb_idx] = fix_monotonicity(
            lower_limit_n[:, nb_idx], upper_limit_n[:, nb_idx])

    return lower_limit_n, upper_limit_n + n_bin_width, n_b


def poissonian_feldman_cousins_interval(n_obs, n_b,
                                        mus,
                                        alpha=0.9,
                                        fix_discrete_n_pathology=False,
                                        n_jobs=1):
    '''
    Calculates confidence belts with the Feldman-Cousins approach.
    For more information see `Feldman-Cousins`_.

    Parameters
    ----------
    n_obs: array-like
        Range of `n_obs` to scan while constructing the limits on
        `n_obs` for each `mu`.
    n_b: array-like
        Number of background events. Parameter for the poissonian
        distribution with background.
    mus: array-like
        Grid of `mu` values to construct the limits on `n_obs` on.
        As `n_b` gets bigger the grid needs to get finer to actually
        populate every important `n_obs` value with an upper limit.
    alpha: float
        The desired confidence level of the constructed confidence belt.
    fix_discrete_n_pathology: bool, optional
        If True, calculate the confidence belts for surrounding n_b to
        correct for a pathology arising from the discreteness of n_obs
        in the poissonian distribution, which causes some upper limits
        to rise with for rising n_b.
    n_jobs: int, optional
        Number of cores to calculate the n_b grid on.

    Returns
    -------
    lower_limit: array-like, shape(len(n_obs), len(n_b))
        The lower limits on `mu` for each `n_obs`.
    upper_limit: array-like, shape(len(n_obs), len(n_b))
        The upper limits on `mu` for each `n_obs`.

    .. _Feldman-Cousins:
        https://arxiv.org/abs/physics/9711021
    '''
    def fix_pathology(arr):
        for j in range(1, len(arr)):
            if arr[-(j + 1)] < arr[-j]:
                arr[-(j + 1)] = arr[-j]
        return arr

    if not isinstance(n_b, Iterable):
        n_b = [n_b]

    if fix_discrete_n_pathology:
        if len(n_b) > 1:
            step_size = n_b[1] - n_b[0]
            if step_size > 0.005:
                warnings.warn(
                    'n_b grid is probably too coarse!')
        else:
            step_size = 0.005
        min_nb = np.min(n_b)
        if np.max(n_b) < min_nb + 1:
            max_nb = np.min(n_b) + 1
        else:
            max_nb = np.max(n_b)
        n_b = np.arange(min_nb, max_nb + step_size, step_size)

    lower_limit_n = np.zeros((len(mus), len(n_b)))
    upper_limit_n = np.zeros((len(mus), len(n_b)))
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for nb in n_b:
            futures.append(
                executor.submit(
                    poissonian_feldman_cousins_acceptance_interval,
                    n_obs=n_obs,
                    n_b=nb,
                    mus=mus,
                    alpha=alpha))
        results = wait(futures)
    for i, future_i in enumerate(results.done):
        lln, uln, nb = future_i.result()
        idx = np.where(n_b == nb)[0][0]
        lower_limit_n[:, idx] = lln.flatten()
        upper_limit_n[:, idx] = uln.flatten()

    # Translate upper and lower limit in measured n into
    # an interval in signal mean mu
    upper_limit_mu = np.zeros((len(n_obs), len(n_b)), dtype=float) - 1.
    lower_limit_mu = np.zeros((len(n_obs), len(n_b)), dtype=float) - 1.

    for n in range(len(n_obs)):
        for b in range(len(n_b)):
            upper_idx = upper_limit_n[:, b] == (n+1)
            lower_idx = lower_limit_n[:, b] == n

            if np.sum(lower_idx) == 0:
                warnings.warn(('The given `mus`-array is probably to coarse.' +
                               ' No upper limit found for `n_obs` = {}. ' +
                               'Setting the upper limit to inf.').format(n))
                upper_limit_mu[n, b] = np.inf
            else:
                upper_limit_mu[n, b] = mus[lower_idx][-1]
            if np.sum(upper_idx) == 0:
                lower_limit_mu[n, b] = 0
            else:
                lower_limit_mu[n, b] = mus[upper_idx][0]

        if fix_discrete_n_pathology:
            lower_limit_mu[n] = fix_pathology(lower_limit_mu[n])
            upper_limit_mu[n] = fix_pathology(upper_limit_mu[n])

    return lower_limit_mu, upper_limit_mu


if __name__ == '__main__':
    import click

    @click.command()
    @click.option('--production', is_flag=True)
    @click.option('--test', is_flag=True)
    def main(production, test):
        if production is False and test is False:
            print('Either enable `production` or `test`, '
                  'otherwise nothing will happen!')

        if production:
            n_obs = np.arange(100)
            mus = np.arange(0, 60.002, 0.002)
            n_b = np.arange(0, 20.002, 0.002)
            auls = average_upper_limit(n_b, n_obs, mus,
                                       alpha=0.9,
                                       fix_discrete_n_pathology=True,
                                       n_jobs=24)
            np.savez('average_upper_limits.npz',
                     n_b=n_b,
                     auls=auls)

        if test:
            from tests.test_feldman_cousins \
                import test_feldman_cousins_intervals
            from tests.test_sensitivity import test_average_upper_limit
            test_feldman_cousins_intervals('tests/fc_intervals_test.pdf')
            test_average_upper_limit('tests/avg_upper_limit_test.pdf')
            print('All tests successful')

        return

    main()
