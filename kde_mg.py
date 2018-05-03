import numpy as np
from multiprocessing import cpu_count, Pool

""" different methods to calculate mean log probability. Each is successfully more optimized at the expense of 
readability"""


def mean_log_prob(data_a, data_b, std_dev):
    """ calculates mean log probability of data_b on data_a
    use numerical stability trick for log(exp(a)+exp(b))
    """
    log_k = np.log(1 / data_a.shape[0])
    std_2 = -2 * np.power(std_dev, 2)
    pi_stddev_2 = np.log(2 * np.pi * np.power(std_dev, 2)) / 2
    summ_m = 0.0
    len_data_a = data_a.shape[0]
    for i in range(data_b.shape[0]):
        summ_k = np.empty(len_data_a)
        for j in range(len_data_a):
            p_x_z = (np.power(data_b[i] - data_a[j], 2) / std_2) - pi_stddev_2
            summ_k[j] = log_k + p_x_z.sum()
        m = np.max(summ_k)
        summ_k = np.exp(summ_k - m)
        lg_prob = np.log(summ_k.sum()) + m
        summ_m += lg_prob

        if i % 1000 == 0:
            print("i: " + str(i))
    return summ_m/data_b.shape[0]


def mean_log_prob_tiling(data_a, data_b, std_dev):
    """ calculates mean log probability of data_b on data_a
    use numerical stability trick for log(exp(a)+exp(b))
    to speed up the summations data_b is duplicated and data_a is subtracted from it.
    """
    log_k = np.log(1 / data_a.shape[0])
    std_2 = -2 * np.power(std_dev, 2)
    pi_stddev_2 = np.log(2 * np.pi * np.power(std_dev, 2)) / 2
    summ_m = 0.0
    for i in range(data_b.shape[0]):
        duplicated_data_b = np.tile(data_b[i], (data_a.shape[0], 1))
        p_x_z = (np.power(duplicated_data_b - data_a, 2) / std_2) - pi_stddev_2
        summ_k = log_k + p_x_z.sum(axis=1)
        m = np.max(summ_k)
        summ_k = summ_k - m
        summ_k = np.exp(summ_k)
        lg_prob = np.log(summ_k.sum()) + m
        summ_m += lg_prob

        if i % 1000 == 0:
            print("i: " + str(i))
    return summ_m/data_b.shape[0]


def mean_log_prob_tiling_multi_proc(data_a, data_b, std_dev):
    """ calculates mean log probability of data_b on data_a
    use numerical stability trick for log(exp(a)+exp(b))
    to speed up the summations data_b is duplicated and data_a is subtracted from it.
    Use multi-processing to assign a core to each inner summation. Noticeable speed up at large problem size.
    """
    pool = Pool(cpu_count())
    log_k = np.log(1 / data_a.shape[0])
    std_2 = -2 * np.power(std_dev, 2)
    pi_stddev_2 = np.log(2 * np.pi * np.power(std_dev, 2)) / 2
    lg_probs_results = [pool.apply_async(inner_summation, (data_a, data_b, i, log_k, pi_stddev_2, std_2,)) for i in range(data_b.shape[0])]
    lg_probs = [result.get() for result in lg_probs_results]
    return np.sum(lg_probs)/data_b.shape[0]


def inner_summation(data_a, data_b, i, log_k, pi_stddev_2, std_2):
    duplicated_data_b = np.tile(data_b[i], (data_a.shape[0], 1))
    p_x_z = (np.power(duplicated_data_b - data_a, 2) / std_2) - pi_stddev_2
    summ_k = log_k + p_x_z.sum(axis=1)
    m = np.max(summ_k)
    summ_k = summ_k - m
    summ_k = np.exp(summ_k)
    lg_prob = np.log(summ_k.sum()) + m
    return lg_prob
