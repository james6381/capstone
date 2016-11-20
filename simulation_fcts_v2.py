import numpy as np
import random
import math

def get_parameters(price_data):
    ret = np.zeros((price_data.size - 1, 1))

    counter = 0
    while counter < price_data.size - 1:  # get returns of the asset
        ret[counter] = np.log((price_data[counter] / price_data[counter + 1])) # NOTE: data is sorted from new to old
        counter += 1

    counter = 0
    mu = 1.
    while counter < ret.size:       # getting geometric average of returns
        mu *= (1+ret[counter])
        counter += 1

    mu **= (1./ret.size)
    mu = np.log(mu**252.)         # converting to annual continuously compounded return

    sigma = (np.var(ret))**0.5 * (252)**0.5       # annualizing standard deviations

    return ret, mu, sigma


def get_1gbmpath(mu1, sig1, mu2, sig2, corr, deltat, t, seeding):
    randmatrix = np.zeros((t, 2))

    random.seed(seeding)
    counter = 0
    while counter < t:  # load in unif(0,1) random numbers in randMatrix
        randmatrix[counter, 0] = random.gauss(0,1)
        randmatrix[counter, 1] = random.gauss(0,1)
        counter += 1

    randmatrix = randmatrix.dot(np.linalg.cholesky(corr).T)  # M = M*chol(corr);
    # generating 2 correlated sets of random numbers
    # using upper triang rather than lower triang matrix, thus Transpose
    # print(np.corrcoef(randmatrix.T))  # checking if method worked

    randmatrix[:,1] *= -1   # due to negative correlations


    randompath = np.ones((2,1))
    counter = 0
    while counter < t:  # simulating the random path for both assets with a geometric brownian motion
        randompath[0] *= np.exp(mu1*deltat + sig1*(deltat**0.5)*randmatrix[counter,0])
        randompath[1] *= np.exp(mu2*deltat + sig2*(deltat**0.5)*randmatrix[counter,1])
        counter += 1

    return randompath


def get_ud(time_step):
    SP500 = np.genfromtxt('SP500_data.csv', delimiter=',')  # need to replace this with backend data
    BND = np.genfromtxt('BND_data.csv', delimiter=',')

    ret_SP500, mu_SP500, sig_SP500 = get_parameters(SP500)
    ret_BND, mu_BND, sig_BND = get_parameters(BND)

    # print(mu_SP500, sig_SP500, mu_BND, sig_BND)

    temp = np.zeros((ret_SP500.size, 2))  # temp matrix for [SP500, BND]
    counter = 0
    while counter < SP500.size - 1:
        temp[counter, 0] = ret_SP500[counter]
        temp[counter, 1] = -1 * ret_BND[counter]  # see note below
        counter += 1

    corrmatrix = np.corrcoef(temp.T)  # for numpy, each row represents a variable, and columns are the observations
    # **NOTE If case SP500 and BND have negative correlation -> matrix will not be positive definite
    # then, multiply the second set of numbers and the correlation by -1?

    #seeding = 10001  # so far, have used seeds 0-10000 # for this function, no need for the seeding; o/w need to add
    # seeding to the counter in the while loop function call
    n = 10000  # number of simulation runs
    results = np.zeros((n, 2))

    counter = 0
    while counter < n:
        temp = get_1gbmpath(mu_SP500, sig_SP500, mu_BND, sig_BND, corrmatrix, 1. / 252., 252 * time_step, counter)
        results[counter, 0] = temp[0]
        results[counter, 1] = temp[1]
        #print('%.8f , %.8f' % (results[counter, 0], results[counter, 1]))  # display results of simulation
        counter += 1

    results[:,0] = np.sort(results[:,0])    # sorting empirical distribution of the assets for that time_step
    results[:,1] = np.sort(results[:,1])

    # get the up, down factors for each scenario and each asset from the empirical dist
    # using math.ceil to ensure first term is an integer
    d1 = results[int(math.ceil(0.25*n)) -1,0]
    u1 = results[int(math.ceil(0.75*n)) -1,0]
    d2 = results[int(math.ceil(0.25*n)) -1,1]
    u2 = results[int(math.ceil(0.75*n)) -1,1]

    return np.array([u1, d1, u2, d2])
