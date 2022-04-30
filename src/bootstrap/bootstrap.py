import numpy as np
from scipy import stats as sts


def bootstrap(stat, data, b=1000):
    t_boot = []
    n = len(data)
    for i in range(b):
        x_stars = np.random.choice(data, n, replace=True)
        t_boot += [stat(x_stars)]
    v_boot = np.var(t_boot)
    se_boot = np.sqrt(v_boot)
    return v_boot, se_boot, t_boot


def pivot_ci(theta_ht, theta_ht_stars, alpha=.05):
    lower_theta_qt = np.quantile(theta_ht_stars, 1 - (alpha / 2))
    upper_theta_qt = np.quantile(theta_ht_stars, alpha / 2)
    a = 2 * theta_ht - lower_theta_qt
    b = 2 * theta_ht - upper_theta_qt
    return a, b


def normal_ci(theta_ht, se_boot, alpha=.05):
    z_alpha2 = sts.norm.ppf(1 - (alpha / 2))
    return theta_ht - z_alpha2 * se_boot, theta_ht + z_alpha2 * se_boot


def percentile_ci(theta_ht_stars, alpha=.05):
    return np.quantile(theta_ht_stars, alpha / 2), np.quantile(theta_ht_stars, 1 - alpha / 2)
