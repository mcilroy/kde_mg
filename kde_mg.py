import numpy as np
from multiprocessing import cpu_count, Pool

""" method to calculate mean log likelihood."""


def mean_log_prob_tiling_multi_proc(data_a, data_b, std_dev):
    """ calculates mean log likelihood of data_b on data_a
    use numerical stability trick for log(exp(a)+exp(b))
    to speed up the summations data_b[i] is duplicated into a matrix and data_a is subtracted from it.
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
