#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt

from ..sensitivity import average_upper_limit


def test_average_upper_limit(filename):
    # Test the average upper limit values from Table XII in the paper.
    # Only calculate the up to 4.0 for the sake of computation time.
    ref_vals_aul_68 = [1.29, 1.52, 1.82, 2.07, 2.29,
                       2.45, 2.62, 2.78, 2.91]
    ref_vals_aul_90 = [2.44, 2.86, 3.28, 3.62, 3.94,
                       4.20, 4.42, 4.63, 4.83]

    n_b = np.arange(0., 4.005, 0.005)
    n_obs = np.arange(101)
    mus = np.arange(0, 30.005, 0.005)
    aul_68 = average_upper_limit(n_b, n_obs, mus,
                                 alpha=0.68,
                                 fix_discrete_n_pathology=True,
                                 n_jobs=2)

    assert np.allclose(np.round(aul_68[::100], decimals=2),
                       ref_vals_aul_68,
                       atol=0.01), \
        'Some of the 68 % average upper limits are wrong! {}'.format(
            np.round(aul_68[::100], decimals=2) - np.asarray(ref_vals_aul_68))

    aul_90 = average_upper_limit(n_b, n_obs, mus,
                                 alpha=0.90,
                                 fix_discrete_n_pathology=True,
                                 n_jobs=2)

    assert np.allclose(np.round(aul_90[::100], decimals=2),
                       ref_vals_aul_90,
                       atol=0.01), \
        'Some of the 90 % average upper limits are wrong! {}'.format(
            np.round(aul_90[::100], decimals=2) - np.asarray(ref_vals_aul_90))

    plt.plot(n_b, aul_68, label=r'$\alpha = 0.68$')
    plt.plot(n_b, aul_90, label=r'$\alpha = 0.90$')
    plt.xlim(0., 4.)
    plt.xlabel(r'$n_b$')
    plt.ylabel(r'Average upper limit')
    plt.legend(loc='best')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return filename


if __name__ == '__main__':
    test_average_upper_limit('avg_upper_limit_test.pdf')
