import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sand.experiments.datasets import data_loader


def get_dataset(dataname, valratio=0.125, batchsize=32, seed=42):
    if dataname == "mice":
        xtrain, xtest, ytrain, ytest, isclassification, numclasses = \
            data_loader.load_mice(seed=seed)

    elif dataname == "isolet":
        xtrain, xtest, ytrain, ytest, isclassification, numclasses = \
            data_loader.load_isolet()

    elif dataname == "activity":
        xtrain, xtest, ytrain, ytest, isclassification, numclasses = \
            data_loader.load_activity()

    elif dataname == "coil":
        xtrain, xtest, ytrain, ytest, isclassification, numclasses = \
            data_loader.load_coil(seed=seed)

    elif dataname == "fashion":
        xtrain, xtest, ytrain, ytest, isclassification, numclasses = \
            data_loader.load_fashion()

    elif dataname == "mnist":
        xtrain, ytrain, xtest, ytest = data_loader.load_mnist()
    
        isclassification = True
        numclasses = 10

    elif dataname == "arcene":
        xtrain, xtest, ytrain, ytest, isclassification, numclasses = \
            data_loader.load_arcene()

    else:
        raise NotImplementedError(f"Unsupported dataset: {dataname}")

    xtrain = np.asarray(xtrain, dtype=np.float32)
    xtest = np.asarray(xtest, dtype=np.float32)
    ytrain = np.asarray(ytrain).reshape(-1)
    ytest = np.asarray(ytest).reshape(-1)

    stratify_labels = ytrain if isclassification else None

    xtr, xval, ytr, yval = train_test_split(
        xtrain,
        ytrain,
        test_size=valratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels
    )

    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr).astype(np.float32)
    xval = scaler.transform(xval).astype(np.float32)
    xtest = scaler.transform(xtest).astype(np.float32)

    if isclassification:
        def transform(x, y):
            x = tf.cast(x, tf.float32)
            y = tf.one_hot(tf.cast(y, tf.int32), depth=numclasses)
            return x, y
    else:
        def transform(x, y):
            return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    dstrain = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    dstrain = (
        dstrain
        .shuffle(buffer_size=len(xtr), seed=seed, reshuffle_each_iteration=True)
        .map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batchsize, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    dsval = tf.data.Dataset.from_tensor_slices((xval, yval))
    dsval = (
        dsval
        .map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batchsize, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    dstest = tf.data.Dataset.from_tensor_slices((xtest, ytest))
    dstest = (
        dstest
        .map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batchsize, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return {
        "xtrain": xtr,
        "ytrain": ytr,
        "xval": xval,
        "yval": yval,
        "xtest": xtest,
        "ytest": ytest,
        "dstrain": dstrain,
        "dsval": dsval,
        "dstest": dstest,
        "numfeatures": xtr.shape[1],
        "isclassification": isclassification,
        "numclasses": numclasses,
        "scaler": scaler,
    }