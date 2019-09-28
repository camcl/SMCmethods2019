import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import pandas as pd
import time

"""
Parameter estimation in the stochastic volatility model
"""

# Read data
datalog = pd.read_csv('~/Documents/PhD/Courses/SMCmethods/OMXLogReturns.csv',
                      header=None,
                      names=['logreturn'])
print('logreturns file contents'.ljust(80, '.'))
print(datalog.describe())
print('\n')
print(datalog.head())
print('\n')


beta = 0.7
phi = 0.98
sigma = 0.16
theta = (phi, sigma, beta)

T = 662
N = 500

initial_volatility = math.log10(3)  # usually a few percents


def softmax(x):
    """Compute the softmax of vector x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def loglikelihood_param(logw):
    """
    log(p(y_1:t) = sum log(sum w_tilde_i) - log N
    """
    t, n = logw.shape
    sum_n = np.sum(np.exp(logw), axis=1)
    return np.sum(np.log10(sum_n) - np.log10(n), axis=0)


def transition(x, phix, sigmax):
    """
    x(t+1) = phi * x(t) + v(t), with v(t)~N(0, sigma^2)
    :param x: x(t)
    :return: x(t+1)
    """
    return np.add(phix * x + stats.norm.rvs(loc=0, scale=sigmax, size=len(x)))


def emission(x, betay):
    """
    y(t) = x(t) + e(t), with e(t)~N(0, B^2 * exp(x(t))
    :param x: x(t)
    :return: y(t)
    """
    return np.add(x + stats.norm.rvs(loc=0, scale=(betay ** 2) * np.exp(x), size=len(x)))


#############################################################################################################
# (a) Particle Metropolis -Hastings for estimating a distributions parameter
#############################################################################################################


def bpf(nb_particles, tau, y_data, theta_vector):
    """
   q is the proposal distribution (normal shape)
   pi is the target distribution
   pi_tilde is pi known up, to scale (normalization factor)
   :param N: number of samples to draw
   :param tau: mean parameter for the proposal distribution
   :param phi: variance parameter for the proposal distribution
   :return:
   """
    print('Bootstrap Particle Filter for N = {} particles, T = {} periods'.format(nb_particles, tau).ljust(80, '.'))
    ph, sg, bt = theta_vector
    # Initialization: sample N initial positions from a chosen distribution, ex N(0, 1)
    x0 = stats.norm.rvs(loc=y_data[0], scale=bt, size=nb_particles)
    w0 = np.repeat(1/nb_particles, repeats=nb_particles)  # all positions are equally weighted i.e. equally likely

    x_t = x0.reshape((1, len(x0)))
    w_t = w0.reshape((1, len(w0)))
    lwt = np.log10(w_t)  #  log-likelihoods
    for t in range(1, tau):
        # Multinomial sampling of ancestor indices from previous weights
        a_t = np.random.choice(range(nb_particles), size=nb_particles, replace=True, p=w_t[t-1, :])

        # Resample
        resampled_x = x_t[t-1, a_t]

        # Generate new x_i from the previous positions, based on the process transition function
        proposed_x = stats.norm.rvs(loc=ph*resampled_x, scale=sg, size=nb_particles)
        x_t = np.vstack((x_t, proposed_x))

        # Update the weights: Compute the new weights based on the observations emission function, and normalize them
        w_tt = stats.norm.logpdf(np.repeat(y_data[t], repeats=nb_particles),
                                 loc=0, scale=bt * np.sqrt(np.exp(x_t[t])))
        w_t = np.vstack((w_t, softmax(w_tt)))
        lwt = np.vstack((lwt, w_tt))

    return np.mean(x_t, axis=1), lwt


def plot_boxlog(nb_particles: int, tau: int, y_data: list) -> None:
    """
    Plot boxplots on log-likelihood of the parameter we want to estimate (phi)
    :param nb_particles:
    :param tau:
    :param y_data:
    :return:
    """
    phis = np.arange(0.08, 1.0, 0.09)
    nb_repet = 10
    df = pd.DataFrame(np.zeros((10, len(phis))), columns=phis)
    tstart = time.time()
    for i, fi in enumerate(phis):
        for rep in range(nb_repet):
            x, lw = bpf(nb_particles, tau, y_data, (fi, sigma, beta))
            df.iloc[rep, i] = loglikelihood_param(lw)
    tstop = time.time()
    print('Total run time = ', tstop-tstart)
    fig, ax = plt.subplots()
    df.boxplot(ax=ax)
    ax.set_xlabel('phi-values')
    ax.set_xticklabels(phis.round(2).tolist())
    ax.set_ylabel('log-likelihood')
    plt.title('Observations log-likelihood for different values of\n'
              'the phi-parameter in the stochastic volatility model')
    plt.savefig('h3a.png')
    plt.show()


# plot_boxlog(N, T, datalog['logreturn'].to_list())


#############################################################################################################
# (b) Particle Metropolis-Hastings for estimating two distributions parameters
#############################################################################################################

def target_likelihood(x, a, b):
    if hasattr(x, 'dtype'): x = x.tolist()
    return stats.invgamma.pdf(x, a, loc=0, scale=math.sqrt(b))


def proposal_likelihood(x, variance):
    return stats.norm.pdf(x, loc=0, scale=math.sqrt(variance))


def target_loglikelihood(x, a, b):
    return stats.invgamma.logpdf(x, a, loc=0, scale=math.sqrt(b))


def proposal_loglikelihood(x, variance):
    return stats.norm.logpdf(x, loc=0, scale=math.sqrt(variance))


def logmean(x):
    return np.log(np.mean(x))


def expmean(x):
    return np.exp(np.mean(x))


def next_state_markov_chain(uniform_prob: object, ratio_lkh: float, params_update: dict) -> dict:
    """
    compute the next state of a Markov Chain process
    :param uniform_prob:
    :param ratio_lkh:
    :param params_update: dict of old/new params values.
    {([0], [1]): []} where [0] is the previous step's value, [1] the newly proposed one
    :return:
    """
    alpha = min(1, ratio_lkh)
    if uniform_prob <= alpha:
        # the new value is more likely than the old and no exploration
        # OR
        # the new value is less likely than the old, but exploration takes over
        for pair, lst in params_update.items():
            if type(pair[1]) is tuple:
                lst.append(np.array(pair[1]))  # convert back immutable tuples as dict.keys to numpy arrays
            else:
                lst.append(pair[1])
    else:
        for pair, lst in params_update.items():
            if type(pair[0]) is tuple:
                lst.append(np.array(pair[0]))
            else:
                lst.append(pair[0])

    return params_update


def mh_sampler(M: int, nb_particles: int, y_data: list, theta_vector: tuple) -> tuple:
    """
    Metroplos-Hastings sampler
    :param M: number of iterations
    :param nb_particles: number of particles to sample at each iteration
    :param y_data: observations data
    :param theta_vector: parameters underlying the target and proposal distributions
    :return:
    """
    # Initialize: set the initial state of the Markov Chain
    np.random.seed(123)
    ph, sg0, bt0 = theta_vector
    x0 = stats.norm.rvs(loc=y_data[0], scale=bt0, size=nb_particles)
    x = [x0]
    sg_density, bt_density = [], []

    sigmas = [sg0]
    betas = [bt0]
    invgamma = {'a': 0.01, 'b': 0.01}

    # Iterate
    for m in range(1, M):
        # Sample sg, bt from the previous values and random walk Gaussian
        sg_rand2, bt_rand2 = -1, -1
        while sg_rand2 < 0 or sg_rand2 > 1:  # avoid sampling negative sigmas
            sg_rand2 = sigmas[m-1] + stats.norm.rvs(loc=0, scale=math.sqrt(0.1))
        while bt_rand2 < 0 or bt_rand2 > 1:
            bt_rand2 = betas[m-1] + stats.norm.rvs(loc=0, scale=math.sqrt(0.1))

        # Calculate a new candidate for the next x value, centered on the previous x value
        xrand = stats.norm.rvs(loc=ph*x[m-1], scale=math.sqrt(sg_rand2))

        # Sample an acceptance threshold between 0 and 1 from uniform distribution:
        # trick that allows for space exploration by the MH algorithm
        u = stats.uniform.rvs(loc=0, scale=1)

        # ratio for accepting the new x value = likelihood new / likelihood old
        # 1.1. Evaluate for beta given y_t
        r_beta = (target_likelihood(y_data[m], invgamma['a'], invgamma['b'])
                  * proposal_likelihood(y_data[m], betas[m-1])) / \
                 (target_likelihood(y_data[m-1], invgamma['a'], invgamma['b'])
                  * proposal_likelihood(y_data[m], bt_rand2))

        # 1.2. Update beta value
        beta_updates = {(betas[m-1], bt_rand2): betas,
                        (target_likelihood(y_data[m-1], invgamma['a'], invgamma['b']),
                         target_likelihood(y_data[m], invgamma['a'], invgamma['b']).tolist()): bt_density
                        }
        beta_updates = next_state_markov_chain(u, r_beta, beta_updates)


        # 2.1. Evaluate for sigma given x_t, and the chosen beta and y_t
        r_sigma = (target_likelihood(np.mean(xrand), invgamma['a'], invgamma['b'])
                   * stats.norm.pdf(y_data[m], loc=0, scale=betas[m]*math.sqrt(expmean(xrand)))  # evaluate likelihood of x given y
                   * proposal_likelihood(np.mean(xrand), sigmas[m-1])) / \
            (target_likelihood(np.mean(x[m-1]), invgamma['a'], invgamma['b'])
             * stats.norm.pdf(y_data[m], loc=0, scale=betas[m]*math.sqrt(expmean(x[m-1])))  # evaluate likelihood of x given y
             * proposal_likelihood(np.mean(xrand), sg_rand2))

        # 1.2. Update sigma value
        sigma_updates = {(tuple(x[m-1]), tuple(xrand)): x,  # tuple() for converting keys to immutable types
                         (sigmas[m-1], sg_rand2): sigmas,
                         (tuple(target_likelihood(x[m-1], invgamma['a'], invgamma['b'])),
                          tuple(target_likelihood(xrand, invgamma['a'], invgamma['b']))): sg_density
                         }
        sigma_updates = next_state_markov_chain(u, r_sigma, sigma_updates)

    return x, betas, sigmas, bt_density, sg_density


def plot_sigma_beta_estimations(M: int, nb_particles: int, y_data: list, theta_vector: tuple) -> None:
    """
    Plot histograms of the marginal distributions approximating parameters of the distributions
    to estimate with a MH sampler
    :param M:
    :param nb_particles:
    :param y_data:
    :param theta_vector:
    :return:
    """
    vol, beta_estim, sigma_estim, beta_marginal, sigma_marginal = mh_sampler(M, nb_particles, y_data, theta_vector)
    print('sigma, beta estimates from PMH --> {}, {}'.format(np.mean(sigma_estim), np.mean(beta_estim)))

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    xlinspace = np.linspace(0, 1, 100)
    ax0.hist(beta_marginal, density=True, label='beta marginal')
    ax0.set_title('beta marginal likelihood given the observations', fontsize=10)
    ax1.hist(sigma_marginal, density=True, label='sigma marginal')
    ax1.set_title('sigma marginal likelihood given the volatility estimate', fontsize=10)
    ax2.plot(xlinspace,
             stats.invgamma.pdf(xlinspace, 0.01, loc=0, scale=math.sqrt(0.01)),
             label='inverse gamma distribution with a = 0.01, b = 0.01')
    ax2.set_title('probability density', fontsize=10)
    plt.legend()
    plt.savefig('h3b.png')
    plt.show()


plot_sigma_beta_estimations(T, N, datalog['logreturn'].to_list(), (0.985, 0.1, 0.8))


#############################################################################################################
# (c) Particle Gibbs sampler
#############################################################################################################

def weighteddiff(x, weights: tuple):
    """
    Returns the diff of successive elements of an array, weighted
    :param x:
    :param weights:
    :return:
    """
    x_i = weights[0] * x[1:]
    x_j = weights[1] * x[:-1]
    return np.subtract(x_i, x_j)


def full_conditional_sigma(x, tau, ph, a, b):
    aprime = a + (tau/2)
    bprime = b + 0.5 * np.power(weighteddiff(x, (1, ph)), 2).sum()
    new_sg2 = stats.invgamma.rvs(aprime, loc=0, scale=np.sqrt(bprime))
    return np.sqrt(new_sg2)


def full_conditional_beta(x, tau, y, a, b):
    aprime = a + (tau/2)
    bprime = b + 0.5 * np.multiply(np.exp(-1*x),
                                   np.power(y, 2)).sum()
    new_bt2 = stats.invgamma.rvs(aprime, loc=0, scale=np.sqrt(bprime))
    return np.sqrt(new_bt2)


def gibbs_sampler(M: int, nb_particles: int, y_data: list, theta_vector: tuple) -> tuple:
    """
    Gibbs sampler: parameters to estimate are refined one after the other at each iteration
    :param M:
    :param nb_particles:
    :param y_data:
    :param theta_vector:
    :return:
    """
    # Initialize: set the initial state of the Markov Chain
    np.random.seed(123)
    ph, sg0, bt0 = theta_vector
    x0 = stats.norm.rvs(loc=y_data[0], scale=bt0, size=nb_particles)
    x = x0.reshape((1, len(x0)))

    sigmas = [sg0]
    betas = [bt0]
    invgamma = {'a': 0.01, 'b': 0.01}

    joint_density = [stats.norm.pdf(y_data[0], loc=0, scale=betas[0]*np.sqrt(np.exp(x[0])))]

    for m in range(1, M):
        # Sample new beta value given x[m-1], y[m-1]
        betas.append(full_conditional_beta(np.mean(x, axis=1), m, y_data[:m], invgamma['a'], invgamma['b']))

        # Evaluate x[m]
        x = np.vstack((x,
                      stats.norm.rvs(loc=ph*x[m-1], scale=sigmas[m-1])))

        # Sample new sigma value
        sigmas.append(full_conditional_sigma(np.mean(x, axis=1), m, ph, invgamma['a'], invgamma['b']))

        # Recompute likelihood
        joint_density.append(stats.norm.pdf(y_data[m], loc=0, scale=betas[m]*np.sqrt(np.exp(x[m, :]))))

    return x, betas, sigmas, joint_density


xgibbs, bt_gibbs, sg_gibbs, density_gibbs = gibbs_sampler(T,
                                                          N,
                                                          np.exp(datalog['logreturn']).to_list(),
                                                          (0.985, 0.1, 0.8))
print(np.mean(bt_gibbs))
print(np.mean(sg_gibbs))
