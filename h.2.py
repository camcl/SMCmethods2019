import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.path import Path as mpath
import math


global A, Q, C, R
A = np.float(0.8)
Q = np.float(0.5)
C = np.float(2)
R = np.float(0.1)

"""
Particle filters implementations for a LGSS model.

Model:
X_t = A * X_t-1 + V_t with V_t ~ N(0, Q) and A = 0.8, Q = 0.5
Y_t = C * X_t + E_t with E_t ~ N(0, R) and C = 2, Q = 0.1
"""

#############################################################################################################
# (a)
#############################################################################################################

np.random.seed(0)
N = 100
T = 2000


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute the softmax of vector x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def true_next_state(x: np.ndarray) -> np.ndarray:
    """Compute the true next particle state"""
    return A * x


def next_state(x: np.ndarray) -> np.ndarray:
    """Compute the next particle state with some noise (process noise)"""
    return A * x + stats.norm.rvs(loc=0, scale=math.sqrt(Q))


def current_observation(x: np.ndarray) -> np.ndarray:
    return C * x + stats.norm.rvs(loc=0, scale=math.sqrt(R))


def random_walk(tau: int) -> tuple:
    """
    Random walk simulation of a particle trajectory
    :param tau: number of iterations for the simulation
    :return: simulated observations, simulated states, and true states
    """
    x0 = stats.norm.rvs(loc=0, scale=math.sqrt(2), size=1)  # no need to sample several particles to simulate
    # the distribution since the analytical expression is known
    y0 = current_observation(x0)
    x = [x0]
    y = [y0]
    ground = [x0]

    for t in range(1, tau):
        x.append(next_state(x[t-1]))
        ground.append(true_next_state(ground[t-1]))
        y.append(current_observation(ground[t]))  # compute noisy oservations from the ground truth

    return y, x, ground


def plot_a(tau: int) -> None:
    """
    Plot the random walk simulation of a particle trajectory
    :param tau: number of iterations for the simulation
    :return:
    """
    ya, xa, ga = random_walk(tau)
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(tau), np.mean(xa, axis=1), label='estimated states', color='tab:olive')
    ax.plot(range(tau), ya, label='observations', color='tab:green')
    ax.plot(range(tau), ga, label='true states', color='tab:orange')
    ax.set_xlabel('Time step')
    ax.set_ylabel('States and observations')
    ax.set_title('R = {}, Q = {}'.format(R, Q))
    plt.suptitle('Random walk simulation of a single particle in a LGSS', fontsize=12)
    plt.legend()
    plt.savefig('h2a.png')
    plt.show()


# plot_a(T)

#############################################################################################################
# (b) Simple case 1D
#############################################################################################################


def apriori_state(x_previous: np.ndarray) -> np.ndarray:
    """Compute the apriori particle state i.e. assuming no process noise"""
    return A * x_previous


def kalman_predict(x_previous: np.ndarray, p_previous: float) -> tuple:
    """
    Implements the prediction step in Kalman filtering
    :param x_previous: mean estimate of the filtering density
    :param p_previous: variance estimate of the filtering density
    :return: next mean estimate with Kalman filtering, and the next variance estimate
    """
    pred_state = apriori_state(x_previous)
    pred_p = A * p_previous * np.transpose(A) + Q

    return pred_state, pred_p


def kalman_update(y: np.ndarray, x_pred: np.ndarray, p_pred: float) -> tuple:
    """
    Implements the updating step in Kalman filtering
    :param y: observation at athe given time step
    :param x_pred: predicted mean of the filtering density
    :param p_pred: predicted variance of the filtering density
    :return: Kalman gain, updated mean and variance estimates of the filtering density
    """
    # K = Kalman's gain: weight the new estimate +/- in relation to the a priori estimate and the measurements
    K = p_pred * np.transpose(C) * np.power(C * p_pred * np.transpose(C) + R, -1)
    x_post = x_pred + K * np.subtract(y, C * x_pred)
    p_post = p_pred - K * C * p_pred

    return K, x_post, p_post


def kalman_filter(tau: int, y_data: np.ndarray, p0: float = 2) -> tuple:
    """
    Recursive algorithm for a Kalman Filter in Linear Gaussian State Space (LGSS) Model
    No random sampling required thanks to the LGSS properties
    1) Predict
    2) Update: 'replace' sampling in other particle filters model used in non linear state-spaces
    :param tau:
    :param p0:
    :param y_data:
    :return:
    """
    print('Kalmans Filter for T = {} periods'.format(tau).ljust(80, '.'))

    # Initialization:
    x0 = 0.0
    k0 = p0 * C * math.pow(C * p0 * C + R, -1)

    x_t = [x0]  # mean of the filtering density at each time step
    p_t = [p0]  # variance of the filtering density at each time step
    k_t = [k0]
    y_t = y_data
    for t in range(1, tau):
        # Prediction step:
        x_pred, p_pred = kalman_predict(x_t[t-1], p_t[t-1])

        # Update step
        k, x_post, p_post = kalman_update(y_t[t], x_pred, p_pred)
        x_t.append(x_post)
        p_t.append(p_post)
        k_t.append(k)

    return x_t, y_t, p_t, k_t


def plot_b(tau: int) -> None:
    """
    Plot particle states estimated from a random walk process
    :param tau: number of iteration steps
    :return:
    """
    ya, xa, ga = random_walk(tau)
    xb, yb, p, k = kalman_filter(tau, ya)
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(tau), xb, label='Kalman estimate', color='b', alpha=0.5)
    ax.plot(range(tau), xa, label='random walk estimate', color='tab:olive', linestyle='dotted')
    ax.plot(range(tau), ga, label='true', color='tab:orange')
    ax.set_xlabel('Time step')
    ax.set_ylabel('States x')
    ax.set_title('Particles states simulations in LGSS with R = {}, Q = {}'.format(R, Q))
    plt.legend()
    plt.savefig('h2b.png')
    plt.show()


# plot_b(T)

#############################################################################################################
# (c) Bootstrap particle filter with multinomial resampling
#############################################################################################################


def bpf(nb_particles: int, tau: int, y_data: np.ndarray, p0: float = 2) -> tuple:
    """
   q is the proposal distribution (normal shape)
   pi is the target distribution
   pi_tilde is pi known up, to scale (normalization factor)
   :param nb_particles: number of samples to draw
   :param tau:
   :param p0:
   :param y_data:
   :return:
   """
    print('Bootstrap Particle Filter for N = {} particles, T = {} periods'.format(nb_particles, tau).ljust(80, '.'))

    # Initialization: sample N initial positions from a chosen distribution, ex N(0, 1)
    x0 = stats.norm.rvs(loc=0, scale=math.sqrt(p0), size=nb_particles)
    w0 = np.repeat(1/nb_particles, repeats=nb_particles)  # all positions are equally weighted i.e. equally likely

    x_t = x0.reshape((1, len(x0)))
    w_t = w0.reshape((1, len(w0)))
    for t in range(1, tau):
        # Multinomial sampling of ancestor indices from previous weights
        a_t = np.random.choice(range(nb_particles), size=nb_particles, replace=True, p=w_t[t-1, :])

        # Resample
        resampled_x = x_t[t-1, a_t]  # Pitfall: same position tends to be selected after several iterations

        # Generate new x_i from the previous positions, based on the process transition function
        proposed_x = stats.norm.rvs(loc=A*resampled_x, scale=math.sqrt(Q), size=nb_particles)
        x_t = np.vstack((x_t, proposed_x))

        # Update the weights: Compute the new weights based on the observations emission function, and normalize them
        w_tt = stats.norm.logpdf(np.repeat(y_data[t], repeats=nb_particles), loc=C*x_t[t], scale=math.sqrt(R))
        w_t = np.vstack((w_t, softmax(w_tt)))

    return np.mean(x_t, axis=1)


def plot_c(nb_particles: int, tau: int) -> None:
    """
    Plot particle states estimated from a random walk process vs. Kalman filtering process
    :param nb_particles: number of particles to sample at each iteration
    :param tau: number of iteration steps
    :return:
    """
    print('\nplot_c function running'.ljust(80, '.'))
    ya, xa, ga = random_walk(tau)
    xb, yb, p, k = kalman_filter(tau, ya)
    xc = bpf(nb_particles, tau, ya)
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(tau), xb, label='kalman', color='b', alpha=0.5)
    ax.plot(range(tau), xc, label='bootstrap', color='r', alpha=0.5)
    ax.plot(range(tau), ga, label='true', color='tab:orange')
    ax.set_xlabel('Time step')
    ax.set_ylabel('States x')
    ax.set_title('Particles states simulations in LGSS with {} particles, R = {}, Q = {}'.format(nb_particles, R, Q))
    plt.legend()
    plt.savefig('h2c.png')
    plt.show()


def table_c(tau: int) -> None:
    """
    Compute approximation error (mean and variance) from a bootstrap particle filtering process vs. Kalman filter
    :param tau: number of iteration steps
    :return:
    """
    print('\ntable_c function running'.ljust(80, '.'))
    ns = [10, 50, 100, 2000, 5000]

    ya, xa, ga = random_walk(tau)
    xb, yb, p, k = kalman_filter(tau, ya)

    series_mean = pd.Series(np.zeros_like(ns, dtype=float))
    series_var = pd.Series(np.zeros_like(ns, dtype=float))

    for i, n in enumerate(ns):
        xc = bpf(n, tau, ya)
        diffmean = np.subtract(xb, xc) # diff for each time step
        diffvar = np.subtract(p, np.var(xc, axis=0)) # p is the variance of the pdf estimated from Kalman filter
        absdiffmean = np.abs(diffmean)
        absdiffvar = np.abs(diffvar)
        mean_absdiff = np.average(absdiffmean)  # average of error mean difference over tau periods
        var_absdiff = np.average(absdiffvar)  # average of error variance difference over tau periods

        series_mean.iloc[i] = np.mean(mean_absdiff)
        series_var.iloc[i] = np.mean(var_absdiff)

    table = pd.DataFrame(data={'nb_particles': ns,
                               'avg_mean': series_mean,
                               'avg_var': series_var})
    print(table)


# plot_c(2000, T)
# table_c(T)

#############################################################################################################
# (d) Fully-adapted particle filter with multinomial resampling
#############################################################################################################


def fapf_multi(nb_particles: int, tau: int, y_data: np.ndarray) -> tuple:
    """
    Fully Adapted Particle Filter = particular case of Auxiliary Particle Filter
    where q and v are known
    q is the proposal distribution
    pi is the target distribution
    pi_tilde is pi known up, to scale (normalization factor)
   :param nb_particles: number of samples to draw. Multinomial resampling
   :param tau:
   :param y_data:
   :return:
   """
    # Initialization: sample N initial positions from a chosen distribution, ex N(0, 1)
    x0 = stats.uniform.rvs(loc=1, scale=math.sqrt(4), size=nb_particles)
    w0 = np.repeat(1/nb_particles, repeats=nb_particles)  # all positions are equally weighted i.e. equally likely
    v0 = np.repeat(1/nb_particles, repeats=nb_particles)

    x_t = x0.reshape((1, len(x0)))
    w_t = w0.reshape((1, len(w0))) # at every step, this is a feature of the fully adaptive settings
    v_t = v0.reshape((1, len(v0)))
    a_t = np.arange(nb_particles)

    for t in range(1, tau):
        # Sample ancestor indices from previous weights
        v = stats.norm.logpdf(np.repeat(y_data[t], repeats=nb_particles), loc=C*x_t[t-1], scale=math.sqrt(R))
        v_t = np.vstack((v_t, softmax(v)))
        a = np.random.choice(range(nb_particles), size=nb_particles, replace=True, p=v_t[t-1])
        a_t = np.vstack((a_t, a))

        # Generate new x_i from the previous positions
        resampled_x = x_t[t-1, a]  # Pitfall: same position tends to be selected after several iteration

        # Propagate
        proposed_x = stats.norm.rvs(loc=A*resampled_x, scale=math.sqrt(Q), size=nb_particles)
        x_t = np.vstack((x_t, proposed_x))

        # Update the weight: Compute the new weights based on the observations emission function, and normalize them
        w_tt = stats.norm.logpdf(np.repeat(y_data[t], repeats=nb_particles), loc=C*x_t[t], scale=math.sqrt(R))
        w_t = np.vstack((w_t, softmax(w_tt)))

    return np.mean(x_t, axis=1), a_t, w_t


def plot_d(nb_particles: int, tau: int) -> None:
    """
    Plot particle states estimated from a fully-adapted particle filtering process
    vs. Kalman filtering process
    :param nb_particles: number of particles to sample at each iteration
    :param tau: number of iteration steps
    :return:
    """
    print('\nplot_d function running'.ljust(80, '.'))
    ya, xa, ga = random_walk(tau)
    xb, yb, p, k = kalman_filter(tau, ya)
    xd, idx, wgs = fapf_multi(nb_particles, tau, ya)
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(tau), xb, label='kalman', color='b', alpha=0.5)
    ax.plot(range(tau), xd, label='fully-adapted', color='r', alpha=0.5)
    ax.plot(range(tau), ga, label='true', color='tab:orange')
    ax.set_xlabel('Time step')
    ax.set_ylabel('States x')
    ax.set_title('Particles states simulations in LGSS with {} particles, R = {}, Q = {}'.format(nb_particles, R, Q))
    plt.legend()
    plt.savefig('h2d.png')
    plt.show()


# plot_d(2000, T)


#############################################################################################################
# (e)
#############################################################################################################


def recurse_particle_path(particle_idx: int, ancestors: np.ndarray, tstart: int, tstop: int,
                          vertices: object = None, codes: object = None) -> tuple:
    """
    Compute recursively the genealogy = path of a given particle
    :param particle_idx: index of the particlein the array
    :param ancestors: array of ancestry particles indices
    :param tstart: time step where to start the genealogy
    :param tstop: time step where to stop finishing looking for new starting paths
    :param vertices: pairs of indices. Associated with the codes, represent the particle path
    :param codes: array of actions to perform on vertices endings (connection type)
    :return: pairs of indices, connection types, color of the path
    """
    # choose 1 particle
    # tstop > tstart
    n = ancestors.shape[1]  # nb of particles
    dye = np.random.rand(3,)
    if vertices is None:
        vertices = []
    if codes is None:
        codes = []

    node = (tstart, particle_idx) # idx in array at time tstart
    childs = []

    if tstart == tstop:
        vertices.append(node)
        vertices.append(node)
        codes.append(mpath.MOVETO)
        codes.append(mpath.LINETO)
    else:
        t = tstart
        for i in range(n):
            if ancestors[t + 1, i] == particle_idx:
                childs.append(i)
                vertices.append(node)
                vertices.append((t + 1, i))
                codes.append(mpath.MOVETO)
                codes.append(mpath.LINETO)
        if t < tstop:
            if len(childs) > 0:
                for i_child in childs:
                    vertices, codes, dye = recurse_particle_path(i_child, ancestors, t + 1, tstop,
                                                            vertices=vertices, codes=codes)

    return vertices, codes, dye


def connectpoints(x: np.ndarray, y: np.ndarray, p1: int, p2: int, particle_color: np.ndarray) -> None:
    """
    Plot segments delimited by dot markers
    :param x: pair of time steps
    :param y: pair of particle indices
    :param p1: start index of the segment to plot
    :param p2: stop index of the segment to plot
    :param particle_color: color for plotting
    :return:
    """
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1, x2], [y1, y2], color=particle_color, marker='.', markersize=2, linestyle='-', linewidth='0.5')


def plot_genealogy_multi(nb_particles: int, tau: int, start: int, stop: int) -> None:
    """
    Plot particles genealogy from a fully-adapted particle filtering process with multinomial resampling
    :param nb_particles:
    :param tau:
    :param start:
    :param stop:
    :return:
    """
    ya, xa, ga = random_walk(tau)
    xd, idx, wgs = fapf_multi(nb_particles, tau, ya)
    for parti in range(nb_particles):
        if parti in [7, 64, 90]:
            print('Path of particle {}'.format(parti).ljust(80, '.'))
            for t in range(start, stop, 1):
                np.random.seed(parti + t)  # randomized choice of color for the path
                vtx, cod, dye = recurse_particle_path(parti, idx, t, tau-1)
                if len(vtx) > 0:
                    path = mpath(vtx, cod)
                    x, y = zip(*path.vertices)
                    for pair in range(0, len(vtx), 2):
                        connectpoints(x, y, pair, pair + 1, dye)
    plt.title('Fully Adapted Particles Filter with multinomial resampling\n'
              'Number of time steps run = {}, Number of particles = {},'
              ' R = {}, Q = {}'.format(tau, nb_particles, R, Q), fontsize=8, va='bottom')
    plt.ylabel('Particle index')
    plt.xlabel('Time step')
    plt.savefig('h2e.png')
    plt.show()


# plot_genealogy_multi(100, T, 1780, 1790)


#############################################################################################################
# (f) FAPF with systematic resampling
#############################################################################################################


def fapf_syst(nb_particles: int, tau: int, y_data: int) -> tuple:
    """
    Fully Adapted Particle Filter = particular case of Auxiliary Particle Filter
    where q and v are known
    q is the proposal distribution
    pi is the target distribution
    pi_tilde is pi known up, to scale (normalization factor)
    Ref for systematic resampling: On Resampling Algorithms for Particle Filters,
    J.D. Hol et al.
   :param N: number of samples to draw. Systematic resampling
   :param tau: number of iterations
   :return: samples approximationg the target distribution, ancestors array, weights array
   """
    # Initialization: sample N initial positions from a chosen distribution, ex N(0, 1)
    x0 = stats.uniform.rvs(loc=1, scale=math.sqrt(4), size=nb_particles)
    w0 = np.repeat(1/nb_particles, repeats=nb_particles)  # all positions are equally weighted i.e. equally likely
    v0 = np.repeat(1/nb_particles, repeats=nb_particles)

    x_t = x0.reshape((1, len(x0)))
    w_t = w0.reshape((1, len(w0))) # at every step, this is a feature of the fully adaptive settings
    v_t = v0.reshape((1, len(v0)))
    a_t = np.ones_like(x_t, dtype=int)

    for t in range(1, tau):
        # Systematic resampling
        v = np.add(np.array(range(nb_particles)) / nb_particles, stats.uniform.rvs(loc=0, scale=1, size=nb_particles))
        v_t = np.vstack((v_t, softmax(v)))
        a = np.random.choice(range(nb_particles), size=nb_particles, replace=True, p=v_t[t-1])
        a_t = np.vstack((a_t, a))

        # Generate new x_i from the previous positions
        resampled_x = x_t[t-1, a]  # Pitfall: same position tends to be selected after several iteration

        # Propagate
        proposed_x = stats.norm.rvs(loc=A*resampled_x, scale=math.sqrt(Q), size=nb_particles)
        x_t = np.vstack((x_t, proposed_x))

        # Update the weight: Compute the new weights based on the observations emission function, and normalize them
        w_tt = stats.norm.logpdf(np.repeat(y_data[t], repeats=nb_particles), loc=C*x_t[t], scale=math.sqrt(R))
        w_t = np.vstack((w_t, softmax(w_tt)))

    return np.mean(x_t, axis=1), a_t, w_t


def plot_genealogy_syst(nb_particles: int, tau: int, start: int, stop: int) -> None:
    """
    Plot particles genealogy from a fully-adapted particle filtering process with systematic resampling
    :param nb_particles:
    :param tau:
    :param start:
    :param stop:
    :return:
    """
    ya, xa, ga = random_walk(tau)
    xd, idx, wgs = fapf_syst(nb_particles, tau, ya)
    for parti in range(nb_particles):
        if parti in [7, 64, 90]:
            print('Path of particle {}'.format(parti).ljust(80, '.'))
            for t in range(start, stop, 1):
                np.random.seed(parti + t)  # randomized choice of color for the path
                vtx, cod, dye = recurse_particle_path(parti, idx, t, tau-1)
                if len(vtx) > 0:
                    path = mpath(vtx, cod)
                    x, y = zip(*path.vertices)
                    for pair in range(0, len(vtx), 2):
                        connectpoints(x, y, pair, pair + 1, dye)
    plt.title('Fully Adapted Particles Filter with systematic resampling\n '
              'Number of time steps run = {}, Number of particles = {},'
              ' R = {}, Q = {}'.format(tau, nb_particles, R, Q), fontsize=8, va='bottom')
    plt.ylabel('Particle index')
    plt.xlabel('Time step')
    plt.savefig('h2f.png')
    plt.show()


# plot_genealogy_syst(100, T, 1780, 1790)


#############################################################################################################
# (g) Make use of the effective population size: Adaptive resampling with ESS trigger
#############################################################################################################


def effective_sample_size(weights: np.ndarray) -> np.float:
    """
    Compute the effective sample size of the sampler i.e. the number of samples from an independent sampler
    that would be needed for achieving the sameapproximation accuracy as with dependent samples
    :param weights:
    :return: effective sample size
    """
    return 1 / np.sum(np.power(weights, 2))


def pf_adaptive_ess(nb_particles: int, tau: int, y_data: np.ndarray) -> tuple:
    """
    Perform fully-adapted particle filtering with adaptive resampling
   :param nb_particles: number of samples to draw.
   :param tau:
   :param y_data:
   :return:
   """
    nthresh = nb_particles / 2
    # Initialization: sample N initial positions from a chosen distribution, ex N(0, 1)
    x0 = stats.uniform.rvs(loc=1, scale=math.sqrt(4), size=nb_particles)
    w0 = np.repeat(1/nb_particles, repeats=nb_particles)  # all positions are equally weighted i.e. equally likely
    counter = 0  # for resampling times

    x_t = x0.reshape((1, len(x0)))
    w_t = w0.reshape((1, len(w0))) # at every step, this is a feature of the fully adaptive settings
    a_t = np.ones_like(x_t, dtype=int)
    neff_t = np.zeros((tau, 1), dtype=float)

    for t in range(1, tau):
        # Compute effective sample size:
        neff = effective_sample_size(w_t[t-1])
        neff_t[t] = neff

        # Adaptive-ESS resampling
        if neff < nthresh:
            counter += 1
            a = np.random.choice(range(nb_particles), size=nb_particles, replace=True, p=w_t[t-1])
            w_t[t-1, :] = w0
        else:
            a = np.arange(nb_particles)
        a_t = np.vstack((a_t, a))

        # Generate new x_i from the previous positions
        resampled_x = x_t[t-1, a]  # Pitfall: same position tends to be selected after several iteration

        # Propagate
        proposed_x = stats.norm.rvs(loc=A*resampled_x, scale=math.sqrt(Q), size=nb_particles)
        x_t = np.vstack((x_t, proposed_x))

        # Update the weight: Compute the new weights based on the observations emission function, and normalize them
        w_tt = stats.norm.logpdf(np.repeat(y_data[t], repeats=nb_particles), loc=C*x_t[t], scale=math.sqrt(R))
        w_t = np.vstack((w_t, softmax(w_tt)))

    return np.mean(x_t, axis=1), a_t, w_t, counter, neff_t


def plot_genealogy_adaptive_ess(nb_particles: int, tau: int, start: int, stop: int) -> None:
    """
    Plot particles path from FAPF with adaptive resampling
    :param nb_particles:
    :param tau:
    :param start:
    :param stop:
    :return:
    """
    ya, xa, ga = random_walk(tau)
    xd, idx, wgs, cnt, neff = pf_adaptive_ess(nb_particles, tau, ya)

    plt.rcParams["figure.figsize"] = [12.4, 4.8]

    for parti in range(nb_particles):
        if parti in [7, 64, 90]:
            print('Path of particle {}'.format(parti).ljust(80, '.'))
            for t in range(start, stop, 1):
                np.random.seed(parti + t)  # randomized choice of color for the path
                vtx, cod, dye = recurse_particle_path(parti, idx, t, tau-1)
                if len(vtx) > 0:
                    path = mpath(vtx, cod)
                    x, y = zip(*path.vertices)
                    for pair in range(0, len(vtx), 2):
                        connectpoints(x, y, pair, pair + 1, dye)
    plt.title('Particles Filter with adaptive resampling\n'
              'Number of time steps run = {}, Number of particles = {},'
              ' R = {}, Q = {}'.format(tau, nb_particles, R, Q), fontsize=8, va='bottom')
    plt.ylabel('Particle index')
    plt.xlabel('Time step')
    plt.savefig('h2g_genealogy.png')
    plt.show()


def plot_neff(nb_particles: int, tau: int) -> None:
    """
    plot effective sample size evolution throughout the filtering process
    :param nb_particles:
    :param tau:
    :return:
    """
    ya, xa, ga = random_walk(tau)
    xd, idx, wgs, cnt, neff = pf_adaptive_ess(nb_particles, tau, ya)

    plt.rcParams["figure.figsize"] = [6.4, 4.8]

    print('Number of resampling events = {}'.format(cnt))

    plt.plot(np.arange(tau), neff, color='k', linewidth=0.7, linestyle='dotted', label='N_eff')
    plt.hlines(nb_particles / 2,xmin=0, xmax=tau, color='b', linewidth=0.7, label='N_threshold')
    plt.xlabel('Time step')
    plt.ylabel('Effective sample size')
    plt.legend()
    plt.savefig('h2g_neff.png')
    plt.show()


plot_genealogy_adaptive_ess(100, T, 1780, 1790)
plot_neff(100, T)  # minimum number of particles resampled is guaranteed
