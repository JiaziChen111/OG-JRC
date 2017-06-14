'''
This module contains functions related to the steady-state

This module defines the following function(s):
    get_SS()

'''

import numpy as np
import scipy.optimize as opt
import os
import household as hh
import firms
import aggregates as aggr
import tax
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def get_SS(args, graph=False):
    (init_vals, beta, sigma, chi_n_vec, l_tilde, b_ellip, upsilon,
        S, alpha, A, delta, etrparam_vec, mtrxparam_vec, mtryparam_vec,
        avg_inc_data) = args
    dist = 10
    mindist = 1e-08
    maxiter = 500
    ss_iter = 0
    xi = 0.2

    r_params = (alpha, A, delta)
    w_params = (alpha, A)
    Y_params = (alpha, A)

    while dist > mindist and ss_iter < maxiter:
        ss_iter += 1
        K, L, X = init_vals
        Y = aggr.get_Y(K, L, Y_params)
        factor = (S * avg_inc_data) / (Y - delta * K)
        r = firms.get_r(K, L, r_params)
        w = firms.get_w(K, L, w_params)
        c1_guess = 0.1
        c1_args = (r, w, X, factor, beta, sigma, chi_n_vec, l_tilde,
                   b_ellip, upsilon, S, etrparam_vec, mtrxparam_vec,
                   mtryparam_vec)
        results_c1 = opt.root(hh.get_bSp1, c1_guess, args=(c1_args))
        c1 = results_c1.x
        cvec, nvec, bvec = hh.get_cnbvecs(c1, c1_args)
        # print('cvec: ', cvec)
        # print('nvec: ', nvec)
        # print('bvec: ', bvec)
        bs_vec = np.append(0, bvec[:-1])
        K_new = max(aggr.get_K(bvec[:-1]), 0.001)
        L_new = max(aggr.get_L(nvec), 0.001)
        lab_inc = w * nvec
        cap_inc = r * bs_vec
        tot_tax_liab_all = tax.get_tot_tax_liab(lab_inc, cap_inc,
                                                factor, etrparam_vec)
        X_new = (1 / S) * (tot_tax_liab_all.sum())
        # factor_new = avg_inc_data / ((1 / S) *
        #                              (r * bs_vec + w * nvec).sum())
        new_vals = np.array([K_new, L_new, X_new])
        dist = ((((new_vals - init_vals) / init_vals) * 100) ** 2).sum()
        init_vals = xi * new_vals + (1 - xi) * init_vals
        print('iter:', ss_iter, ' dist: ', dist)
        print(init_vals)

    c_ss = cvec
    n_ss = nvec
    b_ss = bs_vec
    K_ss = K_new
    L_ss = L_new
    r_ss = r
    w_ss = w
    Y_params = (alpha, A)
    Y_ss = aggr.get_Y(K_ss, L_ss, Y_params)
    C_ss = aggr.get_C(c_ss)
    X_ss = X_new
    factor_ss = factor
    tot_tax_liab_all_ss = tot_tax_liab_all
    tot_tax_liab_ss = X_ss * S

    ss_output = {'c_ss': c_ss, 'n_ss': n_ss, 'b_ss': b_ss, 'K_ss': K_ss,
                 'L_ss': L_ss, 'r_ss': r_ss, 'w_ss': w_ss, 'Y_ss': Y_ss,
                 'C_ss': C_ss, 'X_ss': X_ss, 'factor_ss': factor_ss,
                 'tot_tax_liab_all_ss': tot_tax_liab_all_ss,
                 'tot_tax_liab_ss': tot_tax_liab_ss}

    if graph:
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = 'images'
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot c_ss, n_ss, b_ss
        # Plot steady-state consumption and savings distributions
        age_pers = np.arange(1, S + 1)
        fig, ax = plt.subplots()
        plt.plot(age_pers, c_ss, marker='D', label='Consumption')
        plt.plot(age_pers, np.append(0, b_ss[:-1]), marker='D',
                 label='Savings')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        # plt.title('Steady-state consumption and savings', fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Units of consumption')
        plt.xlim((0, S + 1))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, 'SS_bc')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot steady-state labor supply distributions
        fig, ax = plt.subplots()
        plt.plot(age_pers, n_ss, marker='D', label='Labor supply')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        # plt.title('Steady-state labor supply', fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Labor supply')
        plt.xlim((0, S + 1))
        # plt.ylim((-0.1, 1.15 * (n_ss.max())))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, 'SS_n')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

    return ss_output
