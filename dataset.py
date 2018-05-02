import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set. Inspired from https://gist.github.com/akesling/5358964
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)
    return img, lbl


def display_mnist(images):
    """ Concatenate a series of images into a single image grid. Save image."""
    width = 28
    height = 28
    images = np.reshape(images, [-1, height, width])
    rows = int(math.sqrt(images.shape[0]))
    columns = int(math.sqrt(images.shape[0]))
    single_img = np.zeros((rows*height, columns*width))
    count = 0
    for i in range(rows):
        for j in range(columns):
            single_img[i*height:(i+1)*height, j*width:(j+1)*width] = images[count]
            count += 1
    plt.imshow(single_img, cmap='gray')
    plt.savefig('mnist_sample.png')


def read_mnist_data(path):
    """ Read MNIST images from file path. Shuffle data. Select 10K for train, 10K for val. 
        10K for test from test file. Convert to float32 and scale to 0-1. """
    all_train_images, train_labels = read("training", path)
    perm_idx = np.random.permutation(all_train_images.shape[0])
    all_train_images = all_train_images[perm_idx]
    all_train_images = all_train_images.astype('float32') / 255
    train_images = all_train_images[0:10000]
    val_images = all_train_images[10000:20000]
    test_images, test_labels = read("testing")
    test_images = test_images.astype('float32') / 255
    return train_images, val_images, test_images


def read_cifar100(path):
    """ Read CIFAR100 images from file path. Shuffle data. Select 10K for train, 10K for val. 
    10K for test from test file. Convert to float32 and scale to 0-1. """
    all_train_images = unpickle_data(os.path.join(path, 'train'))
    test_images = unpickle_data(os.path.join(path, 'test'))
    perm_idx = np.random.permutation(all_train_images.shape[0])
    all_train_images = all_train_images[perm_idx]
    all_train_images = all_train_images[0:20000]
    train_images = all_train_images[0:10000]
    train_images = train_images.astype('float32') / 255
    val_images = all_train_images[10000:20000]
    all_train_images = None  # prevent memory errors for my laptop
    val_images = val_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    return train_images, val_images, test_images


def unpickle_data(file):
    """ Load data helper function"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data']


def display_cifar100(images):
    """ Concatenate a series of images into a single image grid. Save image."""
    width = 32
    height = 32
    images = np.reshape(images, [-1, 3, height, width])
    images = np.transpose(images, (0, 2, 3, 1))
    rows = int(math.sqrt(images.shape[0]))
    columns = int(math.sqrt(images.shape[0]))
    single_img = np.zeros((rows * height, columns * width, 3))
    count = 0
    for i in range(rows):
        for j in range(columns):
            single_img[i*height:(i+1)*height, j*width:(j+1)*width, :] = images[count]
            count += 1
    plt.imshow(single_img)
    plt.savefig('cifar100_sample.png')
