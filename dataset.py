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
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
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
    images = np.reshape(images, [-1, 28, 28])
    rows = int(math.sqrt(images.shape[0]))
    columns = int(math.sqrt(images.shape[1]))
    # single_img = np.zeros((rows*28, columns*28))
    # for i in range(rows):
    #     for j in range(columns):
    #         single_img[i:(i+1)*28, i:(i+1)*28] =
    images = np.reshape(images, [rows*28, columns*28])
    plt.imshow(images, cmap='gray')
    plt.show()
    fig, ax = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True)
    count = 0
    for row in range(rows):
        for col in range(columns):
            ax[row, col].imshow(images[count], cmap='gray')
            count += 1
    plt.tight_layout()
    plt.show()


def read_mnist_data(path):
    all_train_images, train_labels = read("training", path)
    perm_idx = np.random.permutation(all_train_images.shape[0])
    all_train_images = all_train_images[perm_idx]
    all_train_images = all_train_images / 255
    train_images = all_train_images[0:10000]
    val_images = all_train_images[10000:20000]
    test_images, test_labels = read("testing")
    test_images = test_images / 255
    return train_images, val_images, test_images


def read_cifar100(path):
    train_dict = unpickle(os.path.join(path, 'train'))
    test_dict = unpickle(os.path.join(path, 'test'))
    all_train_images = train_dict[b'data']
    test_images = test_dict[b'data']
    perm_idx = np.random.permutation(all_train_images.shape[0])
    all_train_images = all_train_images[perm_idx]
    all_train_images = all_train_images[0:20000]
    train_images = all_train_images[0:10000]
    train_images = train_images.astype('float32') / 255
    val_images = all_train_images[10000:20000]
    val_images = val_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    return train_images, val_images, test_images


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def display_cifar100(images):
    images = np.reshape(images, [-1, 3, 32, 32])
    images = np.transpose(images, (0, 2, 3, 1))
    rows = int(math.sqrt(images.shape[0]))
    columns = int(math.sqrt(images.shape[1]))
    fig, ax = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True)
    count = 0
    for row in range(rows):
        for col in range(columns):
            ax[row, col].imshow(images[count])
            count += 1
    plt.tight_layout()
    plt.show()

