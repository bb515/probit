"""Diabetes dataset."""
import numpy as np
import h5py
import warnings
import re
import matplotlib.pyplot as plt
import pathlib
from scipy.fft import fft, rfft
from datetime import datetime
import shelve
import csv
from matplotlib import cm
from scipy.signal import find_peaks
write_path = pathlib.Path("./Diabetes/5bin/")
#write_path = pathlib.Path("./Diabetes/")

def preprocess_comma(line):
    """Separate out columns of 'line' with blank space. Return the columns as a list of strings."""
    # remove the training newline char
    line = re.sub("\n", "", line)
    # remove the surrounding whitespace
    line = line.strip()
    # check of EOF
    if line == '':
        return 1
    cols = re.split(",", line)
    print(cols)
    return cols


def preprocess(line):
    """Separate out columns of 'line' with blank space. Return the columns as a list of strings."""
    # remove the training newline char
    line = re.sub("\n", "", line)
    # remove the surrounding whitespace
    line = line.strip()
    # check of EOF
    if line == '':
        return 1
    cols = re.split("\s+", line)
    return cols


def postprocess_trend(cols, format_string="%Y-%m-%d %H:%M:%S"):
    """
    Return the values corresponding to the IDs in the list of strings.

    A concrete example of columns is below
    cols = [
        2020-01-01          # 0 trend_data_time
        12:23:00            # 1
        179.69232177734375  # 2 trend_data_min
        180.53846740722656  # 3 trend_data_avg
        181.84616088867188  # 4 trend_data_max
    ]
    """
    x_1 = float(cols[0])
    x_2 = float(cols[1])
    #t_n = int(cols[2])
    t_n = float(cols[2])
    return [x_1, x_2], t_n


def read_txt_trend(file_name):
    """Read .txt file and turn the columns into np.ndarray."""
    t = []
    X = []
    with open(file_name, 'r') as f:
        # read until we hit end of file
        while True:
            line = f.readline()
            #cols = preprocess_comma(line)
            cols = preprocess(line)
            if cols == 1:
                # End of file
                break
            values = postprocess_trend(cols)
            print(values)
            data_point, t_n = postprocess_trend(cols)
            X.append(data_point)
            t.append(t_n)
    X = np.array(X)
    t = np.array(t)
    return X, t


def read_partitioned_data(list_of_locations):
    ts = []
    Xs = []
    for file_name in list_of_locations:
        X = []
        t = []
        with open(file_name, 'r') as f:
            # read until we hit end of file
            while True:
                line = f.readline()
                cols = preprocess(line)
                #cols = preprocess_comma(line)
                if cols == 1:
                    # End of file
                    break
                values = postprocess_trend(cols)
                #print(values)
                data_point, t_n = postprocess_trend(cols)
                X.append(data_point)
                t.append(t_n)
        Xs.append(X)
        ts.append(t)
    return np.array(Xs), np.array(ts)

list_of_locations = []
list_of_test_locations = []
for i in range(1, 21):
    list_of_locations.append(write_path / 'diabetes.data_train_5.{}'.format(i))
    list_of_test_locations.append(write_path / 'diabetes.data_test_5.{}'.format(i))

print(list_of_test_locations)
print(list_of_locations)


X_trains, t_trains = read_partitioned_data(list_of_locations)
X_tests, t_tests = read_partitioned_data(list_of_test_locations)

print(np.shape(X_tests))
print(np.shape(t_tests))
print(np.shape(X_trains))
print(np.shape(t_trains))

np.savez(write_path/"data_diabetes.npz", X_test=X_tests, t_test=t_tests, X_train=X_trains, t_train=t_trains)

assert 0

#X, t = read_partitioned_data(list_of_locations)

X, t = read_txt_trend(write_path / 'diabetes.data')

print(X, t)


import sys
np.set_printoptions(threshold=sys.maxsize)
print(X)
print(t)

print(np.shape(X))
print(np.shape(t))

plt.scatter(X[:, 0], X[:, 1])
plt.show()


# plt.scatter(X[np.where(t == 1)][:, 0], X[np.where(t == 1)][:, 1])
# plt.scatter(X[np.where(t == 2)][:, 0], X[np.where(t == 2)][:, 1])
# plt.scatter(X[np.where(t == 3)][:, 0], X[np.where(t == 3)][:, 1])
# plt.scatter(X[np.where(t == 4)][:, 0], X[np.where(t == 4)][:, 1])
# plt.scatter(X[np.where(t == 5)][:, 0], X[np.where(t == 5)][:, 1])
# plt.title('diabetes dataset')
# plt.show()

np.savez(write_path/"data_diabetes_test.npz", X=X, y=t)

