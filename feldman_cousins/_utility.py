#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, print_function

import numpy as np


def fix_monotonicity(lower_limit, upper_limit):
    lower_limit = np.asarray(lower_limit)
    upper_limit = np.asarray(upper_limit)

    mask_l = np.isfinite(lower_limit)
    lower_limit_m = lower_limit[mask_l]
    mask_u = np.isfinite(upper_limit)
    upper_limit_m = upper_limit[mask_u]

    all_fixed = False

    while not all_fixed:
        all_fixed = True
        for i in range(1, len(lower_limit_m)):
            if lower_limit_m[i] < lower_limit_m[i-1]:
                lower_limit_m[i-1] = lower_limit_m[i]
                all_fixed = False
        for j in range(1, len(upper_limit_m)):
            if upper_limit_m[j] < upper_limit_m[j-1]:
                upper_limit_m[j-1] = upper_limit_m[j]
                all_fixed = False

    lower_limit[mask_l] = lower_limit_m
    upper_limit[mask_u] = upper_limit_m

    return lower_limit, upper_limit
