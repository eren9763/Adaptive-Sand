# coding=utf-8
# Copyright 2024 The XXXX-1 Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loader functions to read various tabular datasets."""

import os

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import urllib.request as urllib2 
from pathlib import Path

DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def load_mice(seed=42):
    """
    Load the Mice Protein Expression dataset.

    This loader performs no feature scaling. Scaling is performed later,
    after the train/validation split, using training data only.
    """
    fillingvalue = -100000
    cachefilepath = os.path.join(DATA_DIR, "mice", "Data_Cortex_Nuclear.csv")

    with open(cachefilepath, "r", encoding="UTF-8") as fp:
        x = np.genfromtxt(
            fp.readlines(),
            delimiter=",",
            skip_header=1,
            usecols=range(1, 78),
            filling_values=fillingvalue,
            encoding="UTF-8"
        )

    with open(cachefilepath, "r", encoding="UTF-8") as fp:
        classes = np.genfromtxt(
            fp.readlines(),
            delimiter=",",
            skip_header=1,
            usecols=range(78, 81),
            dtype=None,
            encoding="UTF-8"
        )

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] == fillingvalue:
                same_class_rows = [
                    k for k in range(classes.shape[0])
                    if np.all(classes[i] == classes[k])
                ]

                valid_values = x[same_class_rows, j]
                valid_values = valid_values[valid_values != fillingvalue]

                if valid_values.size > 0:
                    x[i, j] = np.mean(valid_values)
                else:
                    x[i, j] = 0.0

    y = np.zeros(classes.shape[0], dtype=np.uint8)

    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "CS"])):
            if val == label:
                y[i] += 2 ** j

    x = x.astype(np.float32)

    rng = np.random.RandomState(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    x = x[indices]
    y = y[indices]

    split_index = x.shape[0] * 4 // 5

    xtrain = pd.DataFrame(x[:split_index])
    xtest = pd.DataFrame(x[split_index:])

    ytrain = pd.DataFrame(
        y[:split_index],
        dtype=np.int32
    ).iloc[:, 0]

    ytest = pd.DataFrame(
        y[split_index:],
        dtype=np.int32
    ).iloc[:, 0]

    isclassification = True
    numclasses = 8

    print("Data loaded...")
    print("xtrain shape:", xtrain.shape, "ytrain shape:", ytrain.shape)
    print("xtest shape:", xtest.shape, "ytest shape:", ytest.shape)

    return xtrain, xtest, ytrain, ytest, isclassification, numclasses

def load_isolet():
    """
    Load ISOLET without fitting a scaler on train+test jointly.

    Z-score normalization is applied later in dataset.py using only
    the final optimization-training subset.
    """
    cachefilepathtrain = os.path.join(
        DATA_DIR,
        "isolet",
        "isolet1+2+3+4.data"
    )

    cachefilepathtest = os.path.join(
        DATA_DIR,
        "isolet",
        "isolet5.data"
    )

    with open(cachefilepathtrain, "r", encoding="UTF-8") as fp:
        xtrain = np.genfromtxt(
            fp.readlines(),
            delimiter=",",
            usecols=range(0, 617),
            encoding="UTF-8"
        )

    with open(cachefilepathtrain, "r", encoding="UTF-8") as fp:
        ytrain = np.genfromtxt(
            fp.readlines(),
            delimiter=",",
            usecols=617,
            encoding="UTF-8"
        )

    with open(cachefilepathtest, "r", encoding="UTF-8") as fp:
        xtest = np.genfromtxt(
            fp.readlines(),
            delimiter=",",
            usecols=range(0, 617),
            encoding="UTF-8"
        )

    with open(cachefilepathtest, "r", encoding="UTF-8") as fp:
        ytest = np.genfromtxt(
            fp.readlines(),
            delimiter=",",
            usecols=617,
            encoding="UTF-8"
        )

    xtrain = xtrain.astype(np.float32)
    xtest = xtest.astype(np.float32)

    ytrain = (ytrain.astype(np.int32) - 1)
    ytest = (ytest.astype(np.int32) - 1)

    isclassification = True
    numclasses = 26

    xtrain = pd.DataFrame(xtrain)
    xtest = pd.DataFrame(xtest)
    ytrain = pd.DataFrame(ytrain, dtype=np.int32).iloc[:, 0]
    ytest = pd.DataFrame(ytest, dtype=np.int32).iloc[:, 0]

    print("Data loaded...")
    print("xtrain shape:", xtrain.shape, "ytrain shape:", ytrain.shape)
    print("xtest shape:", xtest.shape, "ytest shape:", ytest.shape)

    return xtrain, xtest, ytrain, ytest, isclassification, numclasses
  
  
def load_arcene():

    x_train = pd.read_csv(
        os.path.join(DATA_DIR, "arcene", "arcene_train.data"),
        sep=r"\s+",
        header=None,
    )

    x_test = pd.read_csv(
        os.path.join(DATA_DIR, "arcene", "arcene_valid.data"),
        sep=r"\s+",
        header=None,
    )

    y_train = pd.read_csv(
        os.path.join(DATA_DIR, "arcene", "arcene_train.labels"),
        header=None,
    ).iloc[:, 0]

    y_test = pd.read_csv(
        os.path.join(DATA_DIR, "arcene", "arcene_valid.labels"),
        header=None,
    ).iloc[:, 0]

    y_train = (y_train == 1).astype(np.int32)
    y_test = (y_test == 1).astype(np.int32)

    return x_train, x_test, y_train, y_test, True, 2


def load_coil(seed=42):
    """
    Load COIL-20 without global normalization over train and test samples.

    Feature scaling is performed only after train/validation splitting.
    """
    samples = []

    for class_id in range(1, 21):
        for image_index in range(72):
            imagefilename = f"obj{class_id}__{image_index}.png"
            imagefilepath = os.path.join(
                DATA_DIR,
                "coil",
                "coil-20-proc",
                imagefilename
            )

            with Image.open(imagefilepath) as objimg:
                resized = objimg.resize((20, 20))
                pixelvalues = [float(value) for value in list(resized.getdata())]

            sample = np.asarray(pixelvalues, dtype=np.float32)
            sample = np.append(sample, class_id)
            samples.append(sample)

    samples = np.asarray(samples, dtype=np.float32)

    rng = np.random.RandomState(seed)
    rng.shuffle(samples)

    data = samples[:, :-1].astype(np.float32)
    targets = (samples[:, -1] - 1).astype(np.int64)

    split_index = data.shape[0] * 4 // 5

    xtrain = pd.DataFrame(data[:split_index])
    xtest = pd.DataFrame(data[split_index:])

    ytrain = pd.DataFrame(
        targets[:split_index],
        dtype=np.int32
    ).iloc[:, 0]

    ytest = pd.DataFrame(
        targets[split_index:],
        dtype=np.int32
    ).iloc[:, 0]

    isclassification = True
    numclasses = 20

    print("Data loaded...")
    print("xtrain shape:", xtrain.shape, "ytrain shape:", ytrain.shape)
    print("xtest shape:", xtest.shape, "ytest shape:", ytest.shape)

    return xtrain, xtest, ytrain, ytest, isclassification, numclasses



def load_mnist(fashion=False, digit=None, normalize=False):
    """
    Load MNIST/Fashion-MNIST.

    `normalize` is kept only for backward compatibility and is ignored.
    All scaling is deferred to dataset.py and fitted on training data only.
    """
    if fashion:
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()

    if digit is not None and 0 <= digit <= 9:
        trainmask = (ytrain == digit)
        testmask = (ytest == digit)

        xtrain = xtrain[trainmask]
        ytrain = ytrain[trainmask]

        xtest = xtest[testmask]
        ytest = ytest[testmask]

    xtrain = xtrain.reshape(
        -1,
        xtrain.shape[1] * xtrain.shape[2]
    ).astype(np.float32)

    xtest = xtest.reshape(
        -1,
        xtest.shape[1] * xtest.shape[2]
    ).astype(np.float32)

    return xtrain, ytrain, xtest, ytest

