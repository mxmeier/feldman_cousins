#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt

from ..feldman_cousins import poissonian_feldman_cousins_interval


def test_feldman_cousins_intervals(filename):
    n_b = 3
    n_obs = np.arange(101)
    mus = np.linspace(0, 30, 2401)
    alpha = 0.9
    lower_limits_mu, upper_limits_mu = poissonian_feldman_cousins_interval(
        n_obs=n_obs,
        n_b=n_b,
        mus=mus,
        alpha=alpha,
        fix_discrete_n_pathology=True,
        n_jobs=2)

    lower_limits = lower_limits_mu[:, 200]
    upper_limits = upper_limits_mu[:, 200]

    plt.plot(n_obs, lower_limits, drawstyle='steps-post', color='C0')
    plt.plot(n_obs, upper_limits, drawstyle='steps-post', color='C0')
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.xlabel(r'$n_{\mathrm{obs}}$')
    plt.ylabel(r'$\mu$')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    # Values from the paper to compare with. The values for the numbers
    # given on top can be found in Table IV.
    ref_vals_ll = [0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.15, 0.89, 1.51, 1.88,
                   2.63, 3.04, 4.01, 4.42, 5.50]
    ref_vals_ul = [1.08, 1.88, 3.04, 4.42, 5.60,
                   6.99, 8.47, 9.53, 10.99, 12.30,
                   13.50, 14.81, 16.00, 17.05, 18.50]
    assert np.allclose(np.round(lower_limits[:15], decimals=2),
                       ref_vals_ll,
                       atol=0.01), \
        'Some of the lower limits are wrong! {}'.format(
            lower_limits[:15] - np.asarray(ref_vals_ll))
    assert np.allclose(np.round(upper_limits[:15], decimals=2),
                       ref_vals_ul,
                       atol=0.01), \
        'Some of the upper limits are wrong! {}'.format(
            upper_limits[:15] - np.asarray(ref_vals_ul))
    return filename


if __name__ == '__main__':
    test_feldman_cousins_intervals('fc_intervals_test.pdf')
