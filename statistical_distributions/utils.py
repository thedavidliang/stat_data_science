import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# function to Simulate a Distribution, utilizing the scipy.stats package
def simulate_distribution(
    sample_size: int,
    scipy_object,
    random_seed: int = 48,
    name: str = '',
    support: np.ndarray = None,
    bins: np.ndarray = None,
    figsize: tuple[int,int] = (14,5),
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples from and plots Distribution, using the scipy.stats package. 
    - assumes discrete distributions are non-negative (i.e. don't play around with negative `loc` argument values)
    
    Args:
        sample_size: number of samples to take from distribution
        scipy_object: scipy.stats object for the distribution (eg. scipy.stats.norm/scipy.stats.poisson)
        random_seed: seed for numpy's random number generator
        name: name of distribution to go into the graphs
        support: array of x values over which the distribution is defined
        bins: bins for plotting sample
        figsize: figsize for plot
        **kwargs: any arguments needed for scipy_object (eg. {loc,scale} for normal distribution)
        
    Returns:
        tuple of:
            - numpy ndarray of sampled random variables
            - numpy ndarray of computed support, used for PDF/PMF and CDF
            - numpy ndarray of theoretical PDF/PMF values, across the support
            - numpy ndarray of empirical CDF across support, as computed from sampled variables
            - numpy ndarray of theoretical CDF values, across support
    """
    # draw random variables from distribution
    np.random.seed(random_seed)
    sample = scipy_object.rvs(**kwargs, size=sample_size)

    # generate support from sample, get appropriate PDF (PMF if discrete)
    min_x = sample.min()
    max_x = sample.max()
    if isinstance(sample[0], np.float64):
        x_range = max_x - min_x
        if support is None:
            support = np.arange(min_x - x_range/4, max_x + x_range/4, x_range/200)
        if bins is None:
            bins = np.arange(min_x - x_range/10, max_x + x_range/10, x_range * 0.08)
        pdf_theoretical = scipy_object.pdf(support, **kwargs)
    elif isinstance(sample[0], np.int64):
        if support is None: 
            support = np.arange(0, int(max_x*1.5))
        if bins is None:
            bins = np.arange(-1, max_x+3)
        pdf_theoretical = scipy_object.pmf(support, **kwargs)
    else:
        print('wrong data type encountered in sample')
        return sample, 

    # get CDF of distribution
    cdf_empirical = [(sample <= x).mean() for x in support]
    cdf_theoretical = scipy_object.cdf(support, **kwargs)

    # sample mean/median, plus [approximate] theoretical mean/median
    median_sample = np.median(sample)
    mean_sample = sample.mean()
    median_theoretical = support[(cdf_theoretical <= 0.5).sum()-1]
    mean_theoretical = (support * pdf_theoretical).sum() * (support[1] - support[0])

    # plot PDF and CDF
    prefix_str = name + ' ' if name else ''
    params_str = ','.join([f'{key}={round(value,3) if isinstance(value,float) else value}' for key,value in kwargs.items()] + [f'size={sample_size}'])
    _plot_distribution(
        sample=sample,
        pdf_theoretical=pdf_theoretical,
        cdf_empirical=cdf_empirical,
        cdf_theoretical=cdf_theoretical,
        support=support,
        bins=bins,
        name=prefix_str,
        params_str=params_str,
    )

    return sample, support, pdf_theoretical, cdf_empirical, cdf_theoretical


# function to Simulate a Distribution, utilizing the numpy.random package
def simulate_distribution_numpy(
    sample_size: int,
    np_rng,
    pdf_lambda, 
    random_seed: int = 48,
    name: str = '',
    support: np.ndarray = None,
    bins: np.ndarray = None,
    figsize: tuple[int,int] = (14,5),
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Simulate and plot sample distribution, PDF, CDF against theoretical PDF,CDF, using numpy

    Arguments:
        sample_size: size of sample to take from distribution
        np_rng: np.random object to generate sample (eg. np.random.poisson, np.random.uniform)
        pdf_lambda: function mapping x to PDF 
        random_seed: seed for numpy's random number generator
        name: name of distribution to go into plot titles
        support: set of x values over which distribution is defined
        bins: bins for plotting sample
        figsize: figsize for plot
        **kwargs: any arguments needed for np_rng (eg. {lam} for np.random.poisson)

    Returns:
        tuple of:
            - numpy ndarray of sampled random variables
            - numpy ndarray of computed support, used for PDF/PMF and CDF
            - numpy ndarray of theoretical PDF/PMF values, across the support
            - numpy ndarray of empirical CDF across support, as computed from sampled variables
            - numpy ndarray of theoretical CDF values, across support
    '''

    # sample from distribution
    np.random.seed(random_seed)
    sample = np_rng(size=sample_size, **kwargs)
    min_x, max_x = sample.min(), sample.max()

    # create support, bins if needed
    if isinstance(sample[0], np.float64):
        x_range = max_x - min_x
        if support is None:
            support = np.arange(min_x - x_range/4, max_x + x_range/4, x_range/200)
        if bins is None:
            bins = np.arange(min_x - x_range/10, max_x + x_range/10, x_range * 0.08)
    elif isinstance(sample[0], np.int64):
        if support is None: 
            support = np.arange(0, int(max_x*1.5))
        if bins is None:
            bins = np.arange(-1, max_x+3)

    # get theoretical PDF, and theoretical/empirical CDF
    pdf_theoretical = np.array(list(map(pdf_lambda, support)))
    cdf_theoretical = pdf_theoretical.cumsum() * (support[1] - support[0])
    cdf_empirical = [(sample <= x).mean() for x in support]

    # plot PDF and CDF
    prefix_str = name + ' ' if name else ''
    params_str = ','.join([f'{key}={round(value,3) if isinstance(value,float) else value}' for key,value in kwargs.items()] + [f'size={sample_size}'])
    _plot_distribution(
        sample=sample,
        pdf_theoretical=pdf_theoretical,
        cdf_empirical=cdf_empirical,
        cdf_theoretical=cdf_theoretical,
        support=support,
        bins=bins,
        name=prefix_str,
        params_str=params_str,
    )

    return sample, support, pdf_theoretical, cdf_empirical, cdf_theoretical


# Helper function: plot the distribution, PDF, CDF
def _plot_distribution(
    sample: np.ndarray, 
    pdf_theoretical: np.ndarray, 
    cdf_empirical: np.ndarray,
    cdf_theoretical: np.ndarray,
    support: np.ndarray,
    bins: np.ndarray,
    median_sample: float = None,
    median_theoretical: float = None,
    mean_sample: float = None,
    mean_theoretical: float = None,
    figsize: tuple[int,int] = (14,5),
    name: str = '',
    params_str: str = '',
) -> None:
    # compute statistics if not provided
    if not median_sample:
        median_sample = np.median(sample)
    if not median_theoretical:
        median_theoretical = support[(cdf_theoretical <= 0.5).sum() - 1]
    if not mean_sample:
        mean_sample = sample.mean()
    if not mean_theoretical:
        mean_theoretical = (support * pdf_theoretical).sum() * (support[1] - support[0])

    # create plots
    fig, ax = plt.subplots(1,2, figsize=figsize)
    sns.histplot(sample, bins=bins, ax=ax[0], label='Distr, empirical', stat='density')
    sns.lineplot(x=support, y=pdf_theoretical, ax=ax[0], label='PDF, theoretical', color=sns.color_palette()[1])
    ax[0].axvline(x=median_sample, label='median, sample', color=sns.color_palette()[2])
    ax[0].axvline(x=median_theoretical, label='median, theoretical', color=sns.color_palette()[3])
    ax[0].axvline(x=mean_sample, label='mean, sample', color=sns.color_palette()[4])
    ax[0].axvline(x=mean_theoretical, label='mean, theoretical', color=sns.color_palette()[5])
    ax[0].set_title(name + 'Distribution: ' + params_str)
    ax[0].legend()

    sns.lineplot(x=support, y=cdf_empirical, ax=ax[1], label='CDF, empirical')
    sns.lineplot(x=support, y=cdf_theoretical, ax=ax[1], label='CDF, theoretical')
    ax[1].axvline(x=median_sample, label='median, sample', color=sns.color_palette()[2])
    ax[1].axvline(x=median_theoretical, label='median, theoretical', color=sns.color_palette()[3])
    ax[1].axvline(x=mean_sample, label='mean, sample', color=sns.color_palette()[4])
    ax[1].axvline(x=mean_theoretical, label='mean, theoretical', color=sns.color_palette()[5])
    ax[1].axhline(y=0.5, color=sns.color_palette()[6])

    ax[1].set_title(name + 'CDF: ' + params_str)
    ax[1].legend()

    ax[0].legend()
    plt.show()
    plt.clf()