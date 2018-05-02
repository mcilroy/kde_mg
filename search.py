import dataset
import kde_mg
import numpy as np
import time
import matplotlib.pyplot as plt


def main():
    mnist()
    cifar100()


def mnist():
    mnist_train, mnist_val, mnist_test = dataset.read_mnist_data('.')
    dataset.display_mnist(mnist_train[:200])
    best_std_dev, best_log_probs = find_optimal_value(mnist_train[0:10], mnist_val[0:10])
    print("Best std dev: " + str(best_std_dev) + " achieved with a mean log probability of: " + str(best_log_probs))
    t0 = time.time()
    mean_log_prob = kde_mg.mean_log_prob(mnist_train[0:100], mnist_test[0:100], best_std_dev)
    total_time = time.time() - t0
    print("Calculating mean log probability of test mnist dataset...")
    print("Running time: " + str(total_time) + " seconds")
    print("Mean log probability on mnist test data: " + str(mean_log_prob))


def cifar100():
    cifar100_train, cifar100_val, cifar100_test = dataset.read_cifar100('cifar-100-python')
    dataset.display_cifar100(cifar100_train[:200])
    best_std_dev, best_log_probs = find_optimal_value(cifar100_train[0:10], cifar100_val[0:10])
    print("Best std dev: " + str(best_std_dev) + " achieved with a mean log probability of: " + str(best_log_probs))
    t0 = time.time()
    mean_log_prob = kde_mg.mean_log_prob(cifar100_train[0:100], cifar100_test[0:100], best_std_dev)
    total_time = time.time() - t0
    print("Calculating mean log probability of test CIFAR100 dataset...")
    print("Running time: " + str(total_time) + " seconds")
    print("Mean log probability on CIFAR100 test data: " + str(mean_log_prob))


def find_optimal_value(train, val):
    std_devs = [0.05, 0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    mean_log_probs = []
    for i, std_dev in enumerate(std_devs):
        mean_log_prob = kde_mg.mean_log_prob(train, val, std_dev)
        mean_log_probs.append(mean_log_prob)
        print(i)
    visualization_of_optimal_value(std_devs, mean_log_probs)
    best_log_probs = np.amax(mean_log_probs)
    best_std_dev = std_devs[np.argmax(mean_log_probs)]
    return best_std_dev, best_log_probs


def visualization_of_optimal_value(std_devs, mean_log_probs):
    plt.plot(std_devs, mean_log_probs)
    plt.xlabel("standard deviation")
    plt.ylabel("mean log probability")
    plt.show()

if __name__ == "__main__":
    main()
