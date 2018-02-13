#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, print_function

import warnings
from collections import Iterable
import numpy as np
import scipy.stats as scs

from feldman_cousins import poissonian_feldman_cousins_interval


def get_nobs_weight_range(n_b, weight_thresh=1e-5):
    '''
    Find the range of `n_obs` that contributes significantly to a
    poissonian weighted sum with `n_b` as its mean.
    Parameters
    ----------
    n_b: int or float
        Mean of the poissonian distribution used for the weighting.
    weight_thresh: float, optional
        Threshold for the contributing weights.
    Returns
    -------
    lower_bound: float
        Lower bound to the `n_obs` values contributing to the sum
        more than `weight_thresh`.
    upper_bound: float
        Upper bound to the `n_obs` values contributing to the sum
        more than `weight_thresh`.
    '''
    if n_b == 0:
        return 0., 1.
    sigma = np.sqrt(n_b)
    n_obs_range = (np.maximum(0, n_b - 5 * sigma), n_b + 7 * sigma)
    n_obs_range = np.arange(*n_obs_range)
    poisson_weight = scs.poisson.pmf(n_obs_range, mu=n_b)
    mask = poisson_weight >= weight_thresh
    lower_bound = np.min(n_obs_range[mask])
    upper_bound = np.max(n_obs_range[mask])
    return lower_bound, upper_bound


def average_upper_limit(n_b, n_obs, mus,
                        alpha=0.9,
                        fix_discrete_n_pathology=True,
                        n_jobs=1):
    '''
    Find the average upper limit for a poissonian distribution with background
    via a poissonian weighted sum over all significant `n_obs` values.
    For more information see `Hill-Rawlins`_ formula (4).
    Parameters
    ----------
    n_b: int, float or Iterable
        Mean of the poissonian distribution with background assuming no signal.
    n_obs: array-like
        Range of `n_obs` to scan while constructing the limits on
        `n_obs` for each `mu`.
    mus: array-like
        Grid of `mu` values to contruct the limits on `n_obs` on.
        As `n_b` gets bigger the grid needs to get finer to actually
        populte every important `n_obs` value with an upper limit.
    alpha: float
        The desired confidence level of the constructed confidence belt.
    fix_discrete_n_pathology: bool, optional (default: True)
        If True, calculate the confidence belts for surrounding n_b to
        correct for a pathology arising from the discreteness of n_obs
        in the poissonian distribution, which causes some upper limits
        to rise with for rising n_b.
    n_jobs: int, optional
        Number of cores to calculate the n_b grid on.
    Returns
    -------
    average_upper_limit: array-like
        Average upper limit with the same shape as the given array for `n_b`.
    .. _Hill-Rawlins:
        https://arxiv.org/abs/astro-ph/0209350
    '''
    if not isinstance(n_b, Iterable):
        n_b = [n_b]

    average_upper_limit = np.zeros(len(n_b))

    lower_limits, upper_limits = poissonian_feldman_cousins_interval(
        n_obs=n_obs,
        n_b=n_b,
        mus=mus,
        alpha=alpha,
        fix_discrete_n_pathology=fix_discrete_n_pathology,
        n_jobs=n_jobs)

    for nb_idx, n_bi in enumerate(n_b):
        n_obs_range = get_nobs_weight_range(n_bi)
        n_obs_mask = np.logical_and(n_obs >= n_obs_range[0],
                                    n_obs <= n_obs_range[1])
        n_obsi = n_obs[n_obs_mask]
        upper_limits_i = upper_limits[n_obs_mask, nb_idx]
        if not np.isfinite(np.sum(upper_limits)):
            warnings.warn('For reasonable input values this '
                          'should lead to finite upper limits!')
        poisson_weight = scs.poisson.pmf(n_obsi, mu=n_bi)
        average_upper_limit[nb_idx] = np.sum(upper_limits_i * poisson_weight)

    return average_upper_limit
