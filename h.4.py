import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math

"""
Sequential Monte-Carlo samplers for a bivariate distribution
"""


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of vector x.
    :param x:array of values
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize the vector x.
    :param x:array of values
    """
    return x / np.sum(x, axis=-1)


def softsum(x: np.ndarray) -> np.ndarray:
    """
    Compute the softsum of log-values.
    :param x: array of values
    """
    e_x = np.exp(x - np.max(x))
    return e_x.sum(axis=0)


def unit_square_logpdf(x: np.ndarray) -> np.ndarray:
    """
    Compute the joint log-joint density of a bivariate arrau
    :param x: array of values
    :return: array of log-densities
    """
    x1, x2 = x[0], x[1]
    logpdf = np.add(np.log(x1), np.log(x2))
    logtrigo1 = np.add(
        np.log(np.power(np.cos(x1*math.pi), 2)),
        np.log(np.power(np.sin(x2*3*math.pi), 6))
    )
    logtrigo2 = np.add(
        logtrigo1,
        np.log(np.exp(-30*(np.power(x1, 2) + np.power(x2, 2))))
    )
    u = np.add(logpdf, logtrigo2)
    return u


def unit_square_pdf(x: np.ndarray) -> np.ndarray:
    """
    Compute the joint log-joint density of a bivariate arrau
    :param x: array of values
    :return: array of log-densities
    """
    # pdf has to be computed jointly?
    x1, x2 = x[0], x[1]
    pdf = np.multiply(x1, x2)
    trigo1 = np.multiply(
        np.power(np.cos(x1*math.pi), 2),
        np.power(np.sin(x2*3*math.pi), 6)
    )
    trigo2 = np.multiply(
        trigo1,
        np.exp(-30*(np.power(x1, 2) + np.power(x2, 2)))
    )
    u = np.multiply(pdf, trigo2)
    return u


def gaussian_randwalk_rvs(mn: float, std: float, nb_particles: int) -> np.ndarray:   # Kernel?
    """
    Perform random sampling for a bivariate normal distribution. All dimensions
    are sampled independently from the same distribution.
    :param mn: mean of the sampling distribution
    :param std: standard deviation of the sampling distribution
    :param nb_particles: size of each dimension
    :return: bivariate sample
    """
    # dimensions can be sampled indepenyly
    out = stats.multivariate_normal.rvs(mean=mn, cov=std, size=(2, nb_particles))
    return out


def gaussian_randwalk_logpdf(x: np.ndarray, std: float) -> np.ndarray:
    """
    Compute log-probability densities from a bivariate centered normal distribution.
    :param x: bivariate array. Each column must correspond to one dimension
    :param std: standard deviation of pdf
    :return: bivariate log-densities
    """
    out = stats.multivariate_normal.logpdf(x.transpose(),
                                           mean=np.repeat(0.0, repeats=2),
                                           cov=np.repeat(std, repeats=2))
    return out


def gaussian_randwalk_pdf(x: np.ndarray, std: float) -> np.ndarray:
    """
    Compute probability densities from a bivariate centered normal distribution.
    :param x: bivariate array. Each column must correspond to one dimension
    :param std: standard deviation of pdf
    :return: bivariate densities
    """
    out = stats.multivariate_normal.pdf(x.transpose(),
                                        mean=np.repeat(0.0, repeats=2),
                                        cov=np.repeat(std, repeats=2))
    return out


def effective_sample_size(weights: np.ndarray):
    """
    Compute effective sample size from dependent values samples from the same distribution
    i.e. how many values sampled from a uniform distribution would result in the same mean over the samples
    :param weights:
    :return:
    """
    return 1 / np.sum(np.power(weights, 2))


# def logannealing(x, t, tau):
#     """Likelihood tempering"""
#     return np.add(unit_square_logpdf(x),
#                   (t / tau) * gaussian_randwalk_logpdf(x, 0.02))


def logannealing(target: np.ndarray, x: np.ndarray, t: int, tau: int,
                 p0=stats.multivariate_normal, p0params=([0.0, 0.0], [1, 1])) -> np.ndarray:
    """
    Likelihood tempering: allows for wider exploration of the state space.
    Used for finding good starting value for a SMC sampler.
    Assumes that target_k(x) is proportional to target(x)**(tau_k) * p0(x)**(1-tau_k)
    with tau_k = k/K where K is the total number of annealing steps.
    :param target: probabilities of samples drawn from a target distribution
    :param x: each dimension must correspond to 1 column
    :param t: time step = iteration
    :param tau: total number of iterations
    :param p0: p0 is a distribution where it is possible to sample directly from. Standard Gaussian per default.
    :param p0params: to be chosen in consistency with the physical reality i.e. particles remains in a unit square
    """
    rv = p0(mean=p0params[0], cov=p0params[1])
    temptarget = (1 - (t / tau)) * target
    print('temptarget --> ', temptarget)
    tempp0 = (t / tau) * rv.logpdf(x.transpose())
    print('temp0 --> ', tempp0)
    return np.add(temptarget,
                  tempp0)


def annealing(target: np.ndarray, x: np.ndarray, t: int, tau: int, adapter: float = 1.0,
              p0: object = stats.multivariate_normal, p0params=([0.5, 0.5], [0.1, 0.1])) -> np.ndarray:
    """
    Likelihood tempering: allows for wider exploration of the state space.
    Used for finding good starting value for a SMC sampler.
    Assumes that target_k(x) is proportional to target(x)**(tau_k) * p0(x)**(1-tau_k)
    with tau_k = k/K where K is the total number of annealing steps.
    :param target: probabilities of samples drawn from a target distribution
    :param x: each dimension must correspond to 1 column
    :param t: time step = iteration
    :param tau: total number of iterations
    :param adapter: implements adaptive annealing scheme
    :param p0: p0 is a distribution where it is possible to sample directly from. Standard Gaussian per default.
    :param p0params: to be chosen in consistency with the physical reality i.e. particles remains in a unit square
    """
    rv = p0(mean=p0params[0], cov=p0params[1])
    temptarget = np.power(target, (1 - (t / tau) * adapter))
    tempp0 = np.power(rv.pdf(x.transpose()), (t / tau) * adapter)
    return np.multiply(temptarget, tempp0)


def smc_sampler(nb_particles: int, tau: int, adaptanneal: bool = False) -> tuple:
    """
    Algorithm description from https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf
    :param nb_particles: sampling length
    :param tau: time step
    :return:
    """
    print('SMC sampler running for N = {} particles, T = {} time steps'.format(nb_particles, tau).ljust(80, '.'))
    nthresh = nb_particles * 0.7

    # Initialization: t = 0
    x_t0 = 0.5
    x0 = gaussian_randwalk_rvs(x_t0, 0.001, nb_particles)
    x_t = np.expand_dims(x0, axis=0)

    w0 = np.ones_like(range(nb_particles)) / nb_particles
    w_t = np.expand_dims(w0, axis=0)
    zratio_t = np.ones(tau)

    events = [0]  # for resampling events

    neff_t = np.zeros((tau, 1), dtype=float)
    np.put(neff_t, 0, effective_sample_size(w0))
    a_t = np.zeros((tau, 1, nb_particles), dtype=int)

    # Iterations
    t = 1
    while t < tau:
        print('\nTIME STEP {}'.format(t).ljust(80, '.'))
        print('Tempering step'.ljust(80, '.'))
        if adaptanneal:
            try:
                ad = neff_t[t] / neff_t[t-1]
            except IndexError:
                ad = 1.0
        else:
            ad = 1.0
        # Choose and compute annealing sequence:
        # Use of annealing: find the right distribution to start from/restart exploration of state space
        annealing_prev = annealing(unit_square_pdf(x_t[-1]), x_t[-1], t-1, tau, adapter=ad)
        annealing_tt = annealing(unit_square_pdf(x_t[-1]), x_t[-1], t, tau, adapter=ad)
        # domain-definition consistency adjustment
        annealing_prev = np.clip(annealing_prev, a_min=1e-5, a_max=1.0)
        annealing_tt = np.clip(annealing_tt, a_min=1e-5, a_max=1.0)

        # Compute weights and Z ratio from the previous weights
        lkh = np.divide(annealing_tt,
                        annealing_prev)
        _w_tt = np.multiply(w_t[-1], lkh)

        zratio_tt = np.sum(_w_tt)
        np.put(zratio_t, t, zratio_tt)

        w_tt = np.apply_along_axis(lambda z: normalize(z), arr=_w_tt, axis=-1)
        w_tt = w_tt.reshape((1, nb_particles))
        w_t = np.vstack((w_t, w_tt))

        # Compute effective sample size (from normalized weights?)
        neff = effective_sample_size(w_t[-1])
        neff_t[t] = neff

        # Adaptive-ESS resampling
        if neff < nthresh:
            events.append(1)
            a = np.random.choice(range(nb_particles), size=nb_particles, replace=True, p=w_t[-1, :])
            w_t[-1] = w0
        else:
            events.append(0)
            a = np.arange(nb_particles)
        a = np.expand_dims(a, axis=0)
        np.put(a_t[t], np.arange(nb_particles), a)

        previous_x = x_t[-1]
        resampled_x = np.take_along_axis(previous_x, a, axis=-1)

        # Sample new particles in both dimensions
        x_tt = np.add(resampled_x.squeeze(),
                      gaussian_randwalk_rvs(0.0, 0.02, nb_particles)
                      ).clip(min=0.0, max=1.0)  # unit square location constraint
        x_tt = x_tt.reshape((1, 2, nb_particles))
        # Metropolis-Hastings step: Compute acceptance rate for each particle:
        for i in range(nb_particles):
            xprev = x_t[-1, :, i]
            xprime = x_tt[:, :, i].squeeze()

            ri_numerator = (
                np.multiply(unit_square_pdf(xprime),
                       gaussian_randwalk_pdf(xprev, 0.02)
                       )
            )
            ri_denominator = (
                np.multiply(unit_square_pdf(xprev),
                       gaussian_randwalk_pdf(xprime, 0.02)
                       )
            )
            ri = np.divide(ri_numerator.clip(min=1e-5), ri_denominator)  # avoid by-0 division
            alphai = min(1, ri)

            ui = stats.uniform.rvs(loc=0, scale=1, size=1)
            if ui > alphai:
                np.put(x_tt[-1, :, :], np.array(i), xprev)  # x_t[-1, :,  i]
            # else: keep x_tt[:, :, i]

        x_tt = x_tt.clip(min=0.0, max=1.0)  # constrains particles in the unit square
        x_t = np.vstack((x_t, x_tt))

        t += 1
    zhat = np.cumprod(zratio_t.squeeze())[-1]

    return x_t, w_t, a_t, neff_t, np.asarray(events), zratio_t


def plot_particles_locations(nb_particles: int, tau: int, steps: list, adaptanneal: bool = False) -> None:
    """
    Plot particles locations at different time steps of the SMC sampling process
    :param nb_particles: number of particles
    :param tau: total number of sampling iterations
    :param steps: chosen time steps for plotting particles locations
    :param k: number of annealing steps
    :return:
    """
    positions, weights, indices, effsizes, resamplings, z = smc_sampler(nb_particles, tau, adaptanneal=adaptanneal)

    nb_subplots = (len(steps) // 2) + 1 if len(steps) % 2 != 0 else len(steps) // 2
    steps = np.asarray(steps).reshape((nb_subplots, 2)) if len(steps) % 2 == 0 \
        else np.asarray(steps.append(np.nan)).reshape((nb_subplots, 2))

    fig, ax = plt.subplots(nb_subplots, 2, constrained_layout=True)
    for i in range(steps.shape[0]):
        for j in range(steps.shape[1]):
            if steps[i, j] != np.nan:
                ax[i, j].scatter(positions[steps[i, j], 0, :], positions[steps[i, j], 1, :], s=0.5)
                ax[i, j].set_xbound(lower=0.0, upper=1.0)
                ax[i, j].set_ybound(lower=0.0, upper=1.0)
                ax[i, j].set_xlabel('x1')
                ax[i, j].set_ylabel('x2')
                ax[i, j].set_title('Time step = {}'.format(steps[i, j]), fontsize=10)
    fig.suptitle('Bivariate SMC Sampler on N = {} particles'.format(nb_particles),
                 fontsize=12, y=0.98)
    plt.savefig('h4a.png')
    plt.show()


def plot_effective_sizes(nb_particles: int, tau: int, adaptanneal: bool = False) -> None:
    positions, weights, indices, effsizes, resamplings, z = smc_sampler(nb_particles, tau, adaptanneal=adaptanneal)
    colorconverter = np.vectorize(lambda z: 'tab:orange' if z == 1 else 'tab:blue')
    clr = colorconverter(resamplings)
    resampled = np.ma.array(effsizes, mask=~np.where(resamplings == 1, True, False))
    regular = np.ma.array(effsizes, mask=np.where(resamplings == 1, True, False))

    fig, (ax1) = plt.subplots(1, 1)
    ax1.scatter(np.arange(tau), resampled, s=1, c='tab:orange', label='Resampled steps')
    ax1.scatter(np.arange(tau), regular, s=1, c='tab:blue', label='Not resampled steps')
    plt.hlines(nb_particles*0.7, xmin=0, xmax=tau, color='k', linewidth=0.7, label='N_threshold', linestyles='dotted')
    ax1.set_ylabel('Effective Sample Size')
    ax1.set_xlabel('Time step')
    ax1.set_title('Resampling events occurence in relation to the Effective Sample Size\n'
                  ' while running a Bivariate SMC Sampler on N = {} particles'.format(nb_particles), fontsize=12)
    plt.legend()
    plt.savefig('h4b.png')
    plt.show()


def plot_normalizing_constant(nb_particles: int, tau: int, adaptanneal: bool = False) -> None:
    positions, weights, indices, effsizes, resamplings, z = smc_sampler(nb_particles, tau, adaptanneal=adaptanneal)
    print('\n\n\nZ', z)
    zhat = np.cumprod(z.squeeze())
    print('\nZ final', zhat)

    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(np.arange(tau), z, color='b', linewidth=0.7, label='Z_t-1 / Z_t')
    #plt.hlines(zhat[-1], xmin=0, xmax=tau, color='k', linewidth=0.7, label='final Z estimate', linestyles='dotted')
    ax1.set_ylabel('Z ratios estimates')
    ax1.set_xlabel('Time step')
    ax1.set_title('Estimate of the normalizing constant Z\n'
                  ' while running a Bivariate SMC Sampler on N = {} particles'.format(nb_particles), fontsize=12)
    plt.legend()
    plt.savefig('h4c.png')
    plt.show()


# plot_particles_locations(100, 1000, [0, 1, 439, 999])
# plot_effective_sizes(100, 200)
plot_normalizing_constant(100, 200)
