import numpy as np
import math
from get_yaml import get_profile
from simulation_fcts_v2 import get_ud
from DSPopt import get_DSPopt
#from DSPopt_v2 import get_DSPopt

# # GETTING PROFILE DATA FROM UI
year, goal, goalmin, liab, inc = get_profile()
t = year.size  # get time horizon
t_period = math.ceil(t / (5+1))  # get time period length; can only have 5+1 periods (including time 0)
                                    # o/w problem becomes too large

MM_return = 1.005   # risk free return


# # GROUPING UP AMOUNTS FOR GOALS, INCOMES & LIABILITIES INTO EACH TIME PERIOD STAGE
goal2 = np.zeros((5+1, 1))  # 5+1 rows because 1 for each time period + 1 for time 0
goalmin2 = np.zeros((5+1, 1))
liab2 = np.zeros((5+1, 1))
inc2 = np.zeros((5+1, 1))

period_counter = 0  # we start at row index 1 because initial year has its own row
year_counter = 0
while period_counter < 5+1:
    temp_goal = 0
    temp_goalmin = 0
    temp_liab = 0
    temp_inc = 0
    temp_counter = 0
    while year_counter < period_counter*t_period:  # discounting all values in between periods to the start of a period
        temp_goal += goal[year_counter] / (MM_return**(temp_counter))
        temp_goalmin += goalmin[year_counter] / (MM_return ** (temp_counter))
        temp_liab += liab[year_counter] / (MM_return ** (temp_counter))
        temp_inc += inc[year_counter] / (MM_return ** (temp_counter))
        temp_counter += 1
        year_counter += 1

    goal2[period_counter] = temp_goal   # assigning the total discounted period amounts
    goalmin2[period_counter] = temp_goalmin
    liab2[period_counter] = temp_liab
    inc2[period_counter] = temp_inc
    period_counter += 1


# # RUN SIMULATION TO GET PERIOD UP, DOWN FACTORS FOR STOCKS AND BONDS
ret_scenario = get_ud(t_period)


# # FEEDING INTO 2 OPTIMIZATIONS w/ DIFFERENT RISK COEF AND LIABILITY PENALTY COEF
risk_coef1 = -4.    # ENSURE THESE ARE DOUBLES
liab_coef1 = -10.
optresult1 = get_DSPopt(ret_scenario, goal2, goalmin2, liab2, inc2, risk_coef1)
#optresult1 = get_DSPopt(ret_scenario, goal2, goalmin2, liab2, inc2, risk_coef1, liab_coef1)

risk_coef2 = -10.
liab_coef2 = -15.
optresult2 = get_DSPopt(ret_scenario, goal2, goalmin2, liab2, inc2, risk_coef2)
#optresult2 = get_DSPopt(ret_scenario, goal2, goalmin2, liab2, inc2, risk_coef2, liab_coef2)


# # READING OPTIMIZATION RESULTS
weight1 = np.zeros((optresult1.size*3/5, 3))
goal_perf1 = np.zeros((optresult1.size*2/5, 2))
weight2 = np.zeros((optresult2.size*3/5, 3))
goal_perf2 = np.zeros((optresult2.size*2/5, 2))

counter = 0
row_counter = 0
while counter < optresult1.size*3/5:     # reading the solution vector to get asset allocations
    weight1[row_counter,0] = optresult1[counter]
    weight1[row_counter,1] = optresult1[counter+1]
    weight1[row_counter, 2] = optresult1[counter + 2]

    weight2[row_counter, 0] = optresult2[counter]
    weight2[row_counter, 1] = optresult2[counter + 1]
    weight2[row_counter, 2] = optresult2[counter + 2]

    row_counter += 1
    counter += 3

row_counter = 0
while counter < optresult1.size:    # reading the solution vector to get goal shortfall/excess
    goal_perf1[row_counter, 0] = optresult1[counter]
    goal_perf1[row_counter, 1] = optresult1[counter + 1]

    goal_perf2[row_counter, 0] = optresult2[counter]
    goal_perf2[row_counter, 1] = optresult2[counter + 1]

    row_counter += 1
    counter +=2

# # NEED TO SAVE RESULTS and export to excel