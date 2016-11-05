from cvxopt import *
from cvxopt.modeling import *
import cvxpy
import numpy as np

def DSP_opt(t, s, n, prob_scenario, ret_scenario, goal, goalmin, liab, inc):
    # t will need to determine number of time periods
    # s number of scenarios branches
    # n number of assets
    # probabilities assigned to each scenario

    t_counter = 0
    s_tot = 0   # s_tot number of TOTAL scenarios
    while t_counter < t:
        s_tot += ((s-1)*t_counter+1)**n
        t_counter += 1

    #
    c_in = np.zeros((s_tot,1))  # will need to load values from data
    c_out = np.zeros((s_tot,1))

    t_counter = 0   # assign values from liab and inc to the optimization vectors
    counter = 0
    while t_counter < t:
        s_counter = 0
        while s_counter < s:
            c_in[counter] = inc[t_counter]
            c_out[counter] = liab[t_counter]

            counter += 1
            s_counter += 1
        t_counter += 1

    #
    m1 = np.zeros((s_tot,s_tot*n))  # 0s and 1s to match X's with proper period
    r = np.zeros((s_tot,s_tot*n))   # returns and 0s & 1s
    m2 = np.zeros((s_tot*n, s_tot*n))   # 0s and 1s for > C_out constraint

    stot_counter = 0
    counter = 0
    while stot_counter < s_tot:
        n_counter = 0
        while n_counter < n:
            m1[stot_counter, counter] = 1.   # assigning the M matrix for constraint MX = RX + C_in - C_out (continuity of wealth)
            m2[stot_counter, counter] = 1.   # assigning M matrix for constraint MX >= C_out (always meet liabilities)
            counter += 1
            n_counter += 1
            stot_counter += 1

    #
    stot_counter = 1    # first row of R matrix is 0s
    s_counter = 0
    counter = 0 # represents index of the node at last time period
    while stot_counter < s_tot:
        while s_counter < s:
            n_counter = 0
            while n_counter < n:
                r[stot_counter, n_counter+counter] = ret_scenario[n_counter+counter]
                # assigning the R matrix for constraint MX = RX + C_in - C_out (continuity of wealth)
                # represents branch returns since last time period
                n_counter += 1
                stot_counter += 1
            s_counter += 1
        counter += s*n  # index to next branch of the time period

    #


    # need to get proper matrix sizings


    # test optimization