import os

import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_batch(fpath, label_key="labels"):
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    path = './cifar-10-batches-py/'

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3072, 1), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        (
            batch_x,
            batch_y,
        ) = load_batch(fpath)

        x_train[(i-1) * 10000: i * 10000] = batch_x.reshape(10000, 3072, 1)
        y_train[(i-1) * 10000:i * 10000] = batch_y

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)
    x_test = x_test.reshape(10000, 3072, 1)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    train_data = [(x, y) for x, y in zip(x_train, y_train)]
    test_data = [(x, y) for x, y in zip(x_test, y_test)]

    return train_data, test_data
