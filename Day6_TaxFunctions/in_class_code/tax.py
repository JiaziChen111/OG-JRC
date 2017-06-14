'''
This module defines functions for the tax rates and total tax liability
'''

# Import libraries
import numpy as np


def get_taxrates(lab_inc, cap_inc, factor, params):
    '''
    This function returns either the ETR's, MTRx's, or MTRy's associated
    with the params vector and the vectors of labor and capital income.
    '''
    (A, B, C, D, max_x, min_x, max_y, min_y, shift_x, shift_y, shift,
        phi) = params
    X = np.maximum(0.0, factor * lab_inc)
    Y = np.maximum(0.0, factor * cap_inc)
    # X = factor * lab_inc
    # Y = factor * cap_inc
    X2 = X ** 2
    Y2 = Y ** 2
    ratio_x = (A * X2 + B * X) / (A * X2 + B * X + 1)
    tau_x = (max_x - min_x) * ratio_x + min_x
    ratio_y = (C * Y2 + D * Y) / (C * Y2 + D * Y + 1)
    tau_y = (max_y - min_y) * ratio_y + min_y
    tax_rates = (((tau_x + shift_x) ** phi) * ((tau_y + shift_y) **
                 (1 - phi)) + shift)

    return tax_rates


def get_tot_tax_liab(lab_inc, cap_inc, factor, etr_params):
    ETR = get_taxrates(lab_inc, cap_inc, factor, etr_params)
    tot_inc = lab_inc + cap_inc
    tot_tax_liab = ETR * tot_inc

    return tot_tax_liab
