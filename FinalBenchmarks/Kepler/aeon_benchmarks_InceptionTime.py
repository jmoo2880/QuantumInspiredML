import numpy as np
import pandas as pd
import tensorflow as tf
from aeon.classification.deep_learning import InceptionTimeClassifier
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from aeon.datasets import load_classification

print(f"Num GPUs Available: {tf.config.list_physical_devices('GPU')}")

# Load the data from aeon
X_train_f, y_train_f = load_classification(name="KeplerLightCurves", split="train", return_metadata=False)
X_test_f, y_test_f = load_classification(name="KeplerLightCurves", split="train", return_metadata=False)
X_train = X_train_f.squeeze(axis=1)
y_train = y_train_f.astype(int)
X_test = X_test_f.squeeze(axis=1)
y_test = y_test_f.astype(int)

# Truncate time series to 200 pts
X_train_window = X_train[:, :200]
X_test_window = X_test[:, :200]

# Isolate class 3 and 5 (2 and 4 in Julia)
class_a_tr_idxs = np.argwhere(y_train == 3).flatten()
x_tr_class_a = X_train_window[class_a_tr_idxs, :]
class_b_tr_idxs = np.argwhere(y_train == 5).flatten()
x_tr_class_b = X_train_window[class_b_tr_idxs, :]
X_train_sub = np.vstack([x_tr_class_a, x_tr_class_b])
y_train_sub = np.concatenate([np.ones(len(class_a_tr_idxs), dtype=np.int8), 2*np.ones(len(class_b_tr_idxs), dtype=np.int8)])

class_a_te_idxs = np.argwhere(y_test == 3).flatten()
x_te_class_a = X_test_window[class_a_te_idxs, :]
class_b_te_idxs = np.argwhere(y_test == 5).flatten()
x_te_class_b = X_test_window[class_b_te_idxs, :]
X_test_sub = np.vstack([x_te_class_a, x_te_class_b])
y_test_sub = np.concatenate([np.ones(len(class_a_te_idxs), dtype=np.int8), 2*np.ones(len(class_b_te_idxs), dtype=np.int8)])

# rescale the data
zs = StandardScaler().fit(X_train_sub)
X_train_sub_zs = zs.transform(X_train_sub)
X_test_sub_zs = zs.transform(X_test_sub)

