from cvxopt import *
import numpy as np

def get_DSPopt(ret_scenario, goal, goalmin, liab, inc, risk_coef):
    # each scenario within a time period are equally likely due to choice of ret_scenario
    s = 2   # s number of scenarios branches per asset
    n = 2   # n number of assets not including risk-free asset
    t = 5   # t will need to determine number of time periods

#   Solution vector will be of the for [ Weight in asset 1, W2, W3 ... repeated for number of total scenarios]
#   followed by [amount short of goal, amount above goal .... repeated for number of total scenarios]
#   thus solution vector size is s_tot * ((n + 1) + 2)  = s_tot * 5

    t_counter = 1
    s_tot = 1
    while t_counter < t+1:
        s_tot += s**(t_counter*n)   # s_tot number of TOTAL scenarios
        t_counter += 1


#   GENERATE C_IN, C_OUT, GOAL VECTORS
    c_in = np.zeros((s_tot,1))  # will need to load values from data
    c_out = np.zeros((s_tot,1))
    goal_opt = np.zeros((s_tot,1))
    goalmin_opt = np.zeros((s_tot,1))

    t_counter = 0   # assign values from liab and inc to the optimization vectors; C_in, C_out
    counter = 0
    while t_counter < t+1:
        # note: liab, inc and goal are vectors of form [time1 value, t2 value, t3 value, t4 value, t5 value]
        # here, we are extending those vectors so it can be used in the optimization
        s_counter = 0
        while s_counter < s**(t_counter*n):
            c_in[counter] = inc[t_counter]
            c_out[counter] = liab[t_counter]
            goal_opt[counter] = goal[t_counter]
            goalmin_opt[counter] = goalmin[t_counter]

            counter += 1
            s_counter += 1
        t_counter += 1


#   GENERATING M  matrix; used for M*X = R*X + C_in - C_out (continuity of wealth constraint)
#   also used for M*X > C_out (always meet liabilities constraint); matrix will be 0s and 1s
    m = np.zeros((s_tot,s_tot*5))

    row_counter = 0  # first row corresponds to initial scenario
    col_counter = 0
    while row_counter < s_tot:  # generating M1 matrix, essentially cascading triplets of 1's
        m[row_counter, col_counter] = 1.
        m[row_counter, col_counter+1] = 1.
        m[row_counter, col_counter+2] = 1.
        col_counter += 3
        row_counter += 1


#   GENERATE R MATRIX
    r = np.zeros((s_tot, s_tot*5))   # rows correspond to a scenario return

    t_counter = 1
    row_counter = 1     # first row corresponds to initial scenario; i.e. returns are 1 at that time (no returns)
    col_counter = 0
    while t_counter < t+1:
        # note: ret_scenario = [SP500up factor, SP500 down factor, BND up factor, BND down factor, risk free rate]
        s_counter = 0
        while s_counter < s**(t_counter*n):
            r[row_counter, col_counter] = ret_scenario[0]   # corresponds to S&P500 up, BND up scenario branch
            r[row_counter, col_counter+1] = ret_scenario[2]
            r[row_counter, col_counter+2] = ret_scenario[4]
            row_counter += 1

            r[row_counter, col_counter] = ret_scenario[0]  # corresponds to S&P500 up, BND down scenario branch
            r[row_counter, col_counter + 1] = ret_scenario[3]
            r[row_counter, col_counter + 2] = ret_scenario[4]
            row_counter += 1

            r[row_counter, col_counter] = ret_scenario[1]  # corresponds to S&P500 down, BND up scenario branch
            r[row_counter, col_counter + 1] = ret_scenario[2]
            r[row_counter, col_counter + 2] = ret_scenario[4]
            row_counter += 1

            r[row_counter, col_counter] = ret_scenario[1]  # corresponds to S&P500 down, BND down scenario branch
            r[row_counter, col_counter + 1] = ret_scenario[3]
            r[row_counter, col_counter + 2] = ret_scenario[4]
            row_counter += 1

            s_counter += 4
            col_counter += 3
        t_counter += 1


#   GENERATING BLW (shortfall amount) and ABV (amount above goal) matrices for constraint:
#   R*X + BLW*X - ABV*X = G (goal vector)
    blw = np.zeros((s_tot, s_tot*5))
    abv = np.zeros((s_tot, s_tot*5))

    stot_counter = 0
    col_counter = s_tot*3
    while stot_counter < s_tot:     # essential cascading 1's at the right; (due to formulation of the solution vector)
        blw[stot_counter, col_counter] = 1.
        abv[stot_counter, col_counter +1] = 1.
        col_counter += 2
        stot_counter += 1


#   GENERATING lower bound matrix L for solution vector
#   all X's (amount of wealth in assets) >= 0; if we want to set a minimum amount in each
#   we can simply change lb to that amount; also all BLW and ABV amounts >= 0 by definition of those amounts
    l = np.eye((s_tot*5))
    lb = np.zeros((s_tot*5,1))


#   GENERATING Expectations of the utility (probability space); the way this is set up, all scenarios in each time
#   period are equally likely
    prob = np.zeros((s_tot,1))

    row_counter = 0
    t_counter = 0
    while t_counter < t+1:
        s_counter = 0
        while s_counter < s**(n*t_counter):
            prob[row_counter] = 1./(s**(n*t_counter))
            row_counter += 1
            s_counter += 1
        t_counter += 1

#   SETTING OPTIMIZATION IN PLACE :
#   max [prob]*{risk_coef * [blw]*[X] + [abv]*[X]}      >>> max {risk_coef*[prob]*[blw]+[prob]*[abv]}*[X]
#   st.
#   [r]*[X] + [c_in] - [c_out] = [m]*[X]    >>> ([r]-[m])*[X] = [c_out] - [c_in]
#   [m]*[X] >= [c_out]
#   [r]*[X] + [blw]*[X] - [abv]*[X] = [goal_opt]
#   [l]*[X] > [lb]

#   Need to convert to min c*X: st A*X < b; Aeq*X = beq
    c = -1.*(risk_coef*(prob.T.dot(blw)) + prob.T.dot(abv))     # multiply by -1 to convert to min problem
    a = np.concatenate((-m, -l))    # multiply by -1 to change direction of > sign
    b = np.concatenate((-c_out, -lb))
    aeq = np.concatenate((r-m, r+blw-abv))
    beq = np.concatenate((c_out-c_in, goal_opt))

    c1 = matrix(c.T)  # converting to cvxopt matrices from numpy arrays; NOTE c1 must be of size (n,1), thus transpose
    a1 = matrix(a)      # otherwise will give error below
    b1 = matrix(b)
    aeq1 = matrix(aeq)
    beq1 = matrix(beq)

    #print type(c1)
    #print c1.typecode
    #print c1.size[1]
    # ^ to diagnose "c must be a dense matrix error"; cvxopt code below:
        #if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1:
        #    raise TypeError("'c' must be a dense column matrix")

    sol = solvers.lp(c1, a1, b1, aeq1, beq1)
    return np.array(sol['x'])

