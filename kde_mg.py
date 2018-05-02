import numpy as np


def mean_log_prob(data_a, data_b, std_dev):
    """ calculates mean log probability of data_b on data_a
    use numerical stability trick for log(exp(a)+exp(b))
    """
    log_k = np.log(1 / data_a.shape[0])
    std_2 = -2 * np.power(std_dev, 2)
    pi_stddev_2 = np.log(2 * np.pi * np.power(std_dev, 2)) / 2
    summ_m = 0.0
    for i in range(data_b.shape[0]):
        summ_k = np.empty((data_a.shape[0]))
        for j in range(data_a.shape[0]):
            p_x_z = (np.power(data_b[i] - data_a[j], 2) / std_2) - pi_stddev_2
            summ_k[j] = log_k + p_x_z.sum()
        m = np.max(summ_k)
        summ_k = summ_k - m
        summ_k = np.exp(summ_k)
        lg_prob = np.log(summ_k.sum()) + m
        summ_m += lg_prob
    return summ_m/data_b.shape[0]
