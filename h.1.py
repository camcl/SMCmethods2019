import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math


"""
Importance sampling strategy and normalization constant estimator
"""


def importance_sampler(nb_samples: int, pifunc: object, qfunc: object,
                       piloc: float = 0.0, piscale: float = 1.0,
                       qloc: float = 0.0, qscale: float = 1.0) -> tuple:
    """
    Implement basic sequential importance sampling for estimating a target distribution
    by sampling from a proposal distribution.
    :param nb_samples: number of samples to draw at each iteration for approximating the target distribution
    :param pifunc: target distribution: scipy.stats.<distribution_name> object
    :param qfunc: proposal distribution: scipy.stats.<distribution_name> object
    :param piloc: mean of the target distribution
    :param piscale: standard deviation of the target distribution
    :param qloc: mean of the proposal distribution
    :param qscale: standard deviation of the proposal distribution
    :return: array of samples representing the apriori distribution, weights, corrected distribution
    """
    # Define target and proposal
    target = lambda x: pifunc.pdf(x, loc=piloc, scale=math.sqrt(piscale))
    proposal = lambda x: qfunc.pdf(x, loc=qloc, scale=math.sqrt(qscale))

    # Sample from proposal
    x = qfunc.rvs(loc=qloc, scale=math.sqrt(qscale), size=nb_samples)

    # Define prior and posterior
    prior_tilde = target(x)
    # prior is not accessible

    # Compute weights = likelihood
    w = np.divide(target(x), proposal(x))
    logw = np.subtract(np.log10(target(x)),
                       np.log10(proposal(x)))

    # Simulated target sample, up to scale
    estimates = np.multiply(w, x)
    logest = np.add(logw, np.log10(x))

    return x, w, estimates  # should approximate the target up to scale


def normalizing_constant_estimator(nb_samples: int, pifunc: object, qfunc: object,
                                   piloc: float = 0.0, piscale: float = 1.0,
                                   qloc: float = 0.0, qscale: float = 1.0) -> float:
    """
    Compute the normalizing constant estimator of the approximated distribution.
    :param nb_samples: number of samples to draw at each iteration for approximating the target distribution
    :param pifunc: target distribution: scipy.stats.<distribution_name> object
    :param qfunc: proposal distribution: scipy.stats.<distribution_name> object
    :param piloc: mean of the target distribution
    :param piscale: standard deviation of the target distribution
    :param qloc: mean of the proposal distribution
    :param qscale: standard deviation of the proposal distribution
    :return: normalization constant estimation
    """
    x_i, w_i, pi_tilde_x_i = importance_sampler(nb_samples, pifunc, qfunc,
                                                piloc=piloc, piscale=piscale, qloc=qloc, qscale=qscale)
    z = (1 / nb_samples) * np.sum(np.divide(pi_tilde_x_i,
                                            x_i))
    return z


def plot_estimator_convergence(figname: str, pifunc: object, qfunc: object,
                               piloc: float = 0.0, piscale: float = 1.0,
                               qloc: float = 0.0, qscale: float = 1.0) -> None:
    """
    Plot histogram of the values sampled for approximating target distribution,
    and the normalization constant estimator.
    :param figname: name for saving the plot as an image
    :param pifunc: target distribution: scipy.stats.<distribution_name> object
    :param qfunc: proposal distribution: scipy.stats.<distribution_name> object
    :param piloc: mean of the target distribution
    :param piscale: standard deviation of the target distribution
    :param qloc: mean of the proposal distribution
    :param qscale: standard deviation of the proposal distribution
    :return: normalization constant estimation
    """
    # array of number of samples to use
    N = np.arange(1000, 50000, 5000)
    x = np.linspace(-2, 2, 100)

    fig, (ax0, ax1) = plt.subplots(2, 1)
    z = []
    for n in N:
        x, w, estim = importance_sampler(n, pifunc, qfunc,
                                         piloc=piloc, piscale=piscale, qloc=qloc, qscale=qscale)
        z_hat = normalizing_constant_estimator(n, pifunc, qfunc,
                                               piloc=piloc, piscale=piscale, qloc=qloc, qscale=qscale)
        z.append(z_hat)

        ax0.hist(estim,
                 density=True,
                 histtype='stepfilled',
                 alpha=0.2,
                 bins=30
                 )
    # Compare with the targeted normal distribution
    xx = np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[1], 100)  # auto-adjust x-axis
    ax0.plot(xx,
             pifunc.pdf(xx, loc=0, scale=math.sqrt(1)),
             label='Target: Normal distribution' if figname == 'h1b.png' else 'Target: Cauchy distribution',
             color='k')
    ax0.plot(xx,
             qfunc.pdf(xx, loc=0, scale=math.sqrt(gamma)),
             label='Proposal: Cauchy distribution' if figname == 'h1b.png' else 'Proposal: Normal distribution',
             color='k',
             linestyle='dashed')
    ax1.plot(N,
             z,
             label='Normalization constant estimate',
             c='k')
    ax0.set_xlabel('x value sampled')
    ax0.set_ylabel('density')
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Z estimate')
    ax0.legend()
    ax1.legend()
    plt.savefig(figname)
    plt.show()


gamma = math.sqrt(0.5)

target_b = stats.norm
proposal_b = stats.cauchy
plot_estimator_convergence(target_b, proposal_b, 'h1b.png', piloc=0.0, piscale=1.0, qloc=0.0, qscale=gamma)

target_c = stats.cauchy
proposal_c = stats.norm
plot_estimator_convergence(target_c, proposal_c, 'h1c.png', piloc=0.0, piscale=gamma, qloc=0.0, qscale=1.0)
