#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, print_function

import numpy as np
import scipy.stats as scs

import warnings


def test_coverage_poisson(n_s, n_b, n_obs, lower_limit, upper_limit, N):
    '''
    Test coverage for a given confidence interval of a
    poissonian distribution with background.

    Parameters
    ----------
    n_s : float
        Signal expectation for the poisson distribution.
    n_b : float
        Background expectation for the poisson distribution.
    n_obs : array-like
        Array of observed values to determine the coverage for.
    lower_limit : array-like
        Lower limit for each n_obs value.
    upper_limit : array-like
        Upper limit for each n_obs value.
    N : int
        Number of pseudo experiments.

    Returns
    -------
    coverage : array-like
        Coverage for each n_obs value.
    '''
    # Generate pseudo-experiments
    lmd = n_s + n_b
    exp = np.random.poisson(lmd, size=N)

    # Count the results of the pseudo-experiments
    unique_n, n_cts = np.unique(exp, return_counts=True)

    # For each n_obs given, check if the corresponding
    # confidence interval contains the tested n_s value
    ns_contained = np.zeros_like(n_obs, dtype=bool)
    for i, n_obs_i in enumerate(n_obs):
        if n_s >= lower_limit[i] and n_s <= upper_limit[i]:
            ns_contained[i] = True
        else:
            ns_contained[i] = False

    # Check if all the pseudo-experiment results are covered
    # by the given limits
    not_covered_matter = True
    isin = np.isin(unique_n, n_obs)
    if np.sum(~isin) != 0:
        no_limits = unique_n[~isin]
        # Check if the last valid limit already exlucdes the upcoming
        # ns so they dont contribute to the coverage either way
        print(ns_contained[no_limits[0] - 1])
        if ns_contained[no_limits[0] - 1] == False:
            not_covered_matter = False
        else:
            warnings.warn('The following pseudo-experiment results ' +
                          'do not have corresponding limits given: {}'.format(
                           unique_n[~isin]))


    # Count the number of pseudo experiments contained in
    # the intervals
    n_contained = 0
    for i, unique_n_i in enumerate(unique_n):
        try:
            idx = np.where(n_obs == unique_n_i)[0][0]
        except IndexError:
            if not_covered_matter == False:
                continue
            else:
                raise ValueError('Found an interval that matters, but coverage for it ' +
                                 'could not be evaluated!')
        if ns_contained[idx] == True:
            n_contained += n_cts[i]

    coverage = n_contained / N

    return coverage
