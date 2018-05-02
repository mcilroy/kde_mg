import numpy as np


def mean_log_prob(data_a, data_b, std_dev):
    summ = 0.0
    for i in range(data_b.shape[0]):
        lg_prob = log_prob(data_a, data_b[i], std_dev)
        summ += lg_prob
    return summ/data_b.shape[0]


def log_prob(data_a, example, std_dev):
    #summ = 0.0
    summ_a = []
    for i in range(data_a.shape[0]):
        g_summ = 0.0
        for j in range(data_a.shape[1]):
            mean_sq = np.power(example[j] - data_a[i][j], 2)
            g_summ += -1 * (mean_sq / (2*np.power(std_dev, 2))) - (np.log(2*np.pi*np.power(std_dev, 2))/2)
        blah = np.log(1/data_a.shape[0]) + g_summ
        #summ += np.exp(blah)
        summ_a.append(blah)
        #summ += blah
    m = np.max(blah)
    blah = np.asarray(blah)
    blah = blah - m
    blah = np.exp(blah)
    blah = np.log(np.sum(blah)) + m

    return blah

