import dataset
import kde_mg
import numpy as np
import time
import matplotlib.pyplot as plt

"""
The goal of the program is to calculate the mean log probability of a dataset B on dataset A.
The model is a mixture of gaussians where each gaussian componenet is centered around a data point in A.
The program loads MNIST and CIFAR100, splits them into train, validation and testing then optimizes the choice of 
standard deviation by running the validation on the training. 
Finally the testing dataset's mean log probability is calculated on the training. 
"""


def main():
    mnist()
    cifar100()


def mnist():
    """ Load MNIST data. Optimize the standard deviation. Calculate mean log probability of the mnist testing dataset. 
    Benchmarked running time."""
    mnist_train, mnist_val, mnist_test = dataset.read_mnist_data('.')
    dataset.display_mnist(mnist_train[:200])
    best_std_dev, best_log_probs = find_optimal_value(mnist_train, mnist_val, "mnist")
    print("Best std dev: " + str(best_std_dev) + " achieved with a mean log probability of: " + str(best_log_probs))
    t0 = time.time()
    mean_log_prob = kde_mg.mean_log_prob_tiling_multi_proc(mnist_train, mnist_test, best_std_dev)
    total_time = time.time() - t0
    print("Calculating mean log probability of test mnist dataset...")
    print("Running time: " + str(total_time) + " seconds")
    print("Mean log probability on mnist test data: " + str(mean_log_prob))


def cifar100():
    """ Load CIFAR100 data. Optimize the standard deviation. Calculate mean log probability of the CIFAR100 testing 
    dataset. Benchmarked running time."""
    cifar100_train, cifar100_val, cifar100_test = dataset.read_cifar100('cifar-100-python')
    dataset.display_cifar100(cifar100_train[:200])
    best_std_dev, best_log_probs = find_optimal_value(cifar100_train, cifar100_val, "cifar100")
    print("Best std dev: " + str(best_std_dev) + " achieved with a mean log probability of: " + str(best_log_probs))
    t0 = time.time()
    mean_log_prob = kde_mg.mean_log_prob_tiling_multi_proc(cifar100_train, cifar100_test, best_std_dev)
    total_time = time.time() - t0
    print("Calculating mean log probability of test CIFAR100 dataset...")
    print("Running time: " + str(total_time) + " seconds")
    print("Mean log probability on CIFAR100 test data: " + str(mean_log_prob))


def find_optimal_value(train, val, dataset_name):
    """ For each std dev calculate mean log probability for validation on training. Select std dev with highest 
    log probability."""
    std_devs = [0.05, 0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    mean_log_probs = []
    for i, std_dev in enumerate(std_devs):
        mean_log_prob = kde_mg.mean_log_prob_tiling_multi_proc(train, val, std_dev)
        mean_log_probs.append(mean_log_prob)
        print(dataset_name + " - finding optimal value % complete: " + str((float(i)/len(std_devs))*100))
    visualization_of_optimal_value(std_devs, mean_log_probs, dataset_name)
    best_log_probs = np.amax(mean_log_probs)
    best_std_dev = std_devs[np.argmax(mean_log_probs)]
    return best_std_dev, best_log_probs


def visualization_of_optimal_value(std_devs, mean_log_probs, dataset_name):
    """ Visualize optimal value with a plot of the different log probabilities for each std dev."""
    plt.figure()
    plt.plot(std_devs, mean_log_probs)
    plt.xlabel("standard deviation")
    plt.ylabel("mean log probability")
    plt.savefig(dataset_name+'_optimal_value_curve.png')


if __name__ == "__main__":
    main()
