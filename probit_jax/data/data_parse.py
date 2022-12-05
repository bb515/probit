"""Diabetes dataset."""
import numpy as np
import h5py
import re
import matplotlib.pyplot as plt
import pathlib
import shelve
import csv
import sys
np.set_printoptions(threshold=sys.maxsize)

list_of_5bins_test_write_paths = [
    "./abalone/5bins/abalone_test_5.",
    "./Auto-Mpg/5bins/auto.data_test_5.",
    "./bostonhousing/5bins/housing_test_5.",
    "./Diabetes/5bins/diabetes.data_test_5.",
    "./machinecpu/5bins/machine_test_5.",
    "./pyrimidines/5bins/pyrim_test_5.",
    "./stocksdomain/5bins/stock_test_5.",
    "./triazines/5bins/triazines_test_5.",
    "./wisconsin/5bins/wpbc_test_5.",
]

list_of_5bins_train_write_paths = [
    "./abalone/5bins/abalone_train_5.",
    "./Auto-Mpg/5bins/auto.data_train_5.",
    "./bostonhousing/5bins/housing_train_5.",
    "./Diabetes/5bins/diabetes.data_train_5.",
    "./machinecpu/5bins/machine_train_5.",
    "./pyrimidines/5bins/pyrim_train_5.",
    "./stocksdomain/5bins/stock_train_5.",
    "./triazines/5bins/triazines_train_5.",
    "./wisconsin/5bins/wpbc_train_5.",
]

list_of_10bins_test_write_paths = [
    "./abalone/10bins/abalone_test_10.",
    "./Auto-Mpg/10bins/auto.data_test_10.",
    "./bostonhousing/10bins/housing_test_10.",
    "./Diabetes/10bins/diabetes.data_test_10.",
    "./machinecpu/10bins/machine_test_10.",
    "./pyrimidines/10bins/pyrim_test_10.",
    "./stocksdomain/10bins/stock_test_10.",
    "./triazines/10bins/triazines_test_10.",
    "./wisconsin/10bins/wpbc_test_10.",
]

list_of_10bins_train_write_paths = [
    "./abalone/10bins/abalone_train_10.",
    "./Auto-Mpg/10bins/auto.data_train_10.",
    "./bostonhousing/10bins/housing_train_10.",
    "./Diabetes/10bins/diabetes.data_train_10.",
    "./machinecpu/10bins/machine_train_10.",
    "./pyrimidines/10bins/pyrim_train_10.",
    "./stocksdomain/10bins/stock_train_10.",
    "./triazines/10bins/triazines_train_10.",
    "./wisconsin/10bins/wpbc_train_10.",
]

list_of_10bins_write_paths = [
    "./abalone/10bins/abalone",
    "./Auto-Mpg/10bins/auto.data",
    "./bostonhousing/10bins/housing",
    "./Diabetes/10bins/diabetes.data.ord",
    "./machinecpu/10bins/machine",
    "./pyrimidines/10bins/pyrim",
    "./stocksdomain/10bins/stock",
    "./triazines/10bins/triazines",
    "./wisconsin/10bins/wpbc",
]

list_of_5bins_write_paths = [
    "./abalone/5bins/abalone",
    "./Auto-Mpg/5bins/auto.data",
    "./bostonhousing/5bins/housing",
    "./Diabetes/5bins/diabetes.data",
    "./machinecpu/5bins/machine",
    "./pyrimidines/5bins/pyrim",
    "./stocksdomain/5bins/stock",
    "./triazines/5bins/triazines",
    "./wisconsin/5bins/wpbc",
]

list_of_write_paths = [
    "./Diabetes/diabetes.DATA",
    "./abalone/abalone",  # Try .File
    "./Auto-Mpg/auto.DATA",
    "./bostonhousing/housing",  #.File
    "./machinecpu/machine",  #.File
    "./pyrimidines/pyrim",  #.File -- looks nice
    "./stocksdomain/stock",  #.File -- looks nice... what is this data?
    "./triazines/triazines",  #.File -- looks nice
    "./wisconsin/wpbc"  #.File -- nice and dense but integer target - fine.
]


# list_of_5bin_write_paths = [
#     "./Diabetes/5bins/",
#     "./abalone/5bins/",
#     "./Auto-Mpg/5bins/",
#     "./bostonhousing/5bins/",
#     "./machinecpu/5bins/",
#     "./pyrimidines/5bins/",
#     "./stocksdomain/5bins/",
#     "./triazines/5bins/",
#     "./wisconsin/5bins/"
# ]
#
# list_of_10bin_write_paths = [
#     "./Diabetes/10bins/",
#     "./abalone/10bins/",
#     "./Auto-Mpg/10bin/",
#     "./bostonhousing/10bins/",
#     "./machinecpu/10bins/",
#     "./pyrimidines/10bins/",
#     "./stocksdomain/10bins/",
#     "./triazines/10bins/",
#     "./wisconsin/10bins/"
# ]


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
    ...
    ]
    """
    xs = []
    for i in range(len(cols) - 1):
        xs.append(float(cols[i]))
    t_n = float(cols[-1])
    return xs, t_n


def read_txt_trend(file_name):
    """Read .txt file and turn the columns into np.ndarray."""
    t = []
    X = []
    with open(file_name, 'r') as f:
        # read until we hit end of file
        while True:
            line = f.readline()
            cols = preprocess_comma(line)
            cols2 = preprocess(line)
            if cols == 1:
                # End of file
                break
            try:
                values = postprocess_trend(cols)
            except:
                values = postprocess_trend(cols2)
            data_point, t_n = values
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


def read_all_datasets(path, list_of_writepaths, folds=20):
    """
    list_of_writepaths
    :arg list_of_writepaths: e.g. ["./abalone/10bins/abalone_train_10.", ...]
    :arg bins: 5 or 10
    :arg folds: 20
    :return: save the datasets in a npz filenp.savez(write_path, X_test=X_tests, t_test=t_tests, X_train=X_trains, t_train=t_trains)
    """
    for write_path in list_of_writepaths:
        print(str(write_path))
        list_of_write_paths_10test = []
        list_of_write_paths_10train = []
        # list_of_write_paths_5test = []
        # list_of_write_paths_5train = []
        for i in range(1, int(1+folds)):
            list_of_write_paths_10train.append(path / (write_path + '_train_10.{}'.format(i)))
            list_of_write_paths_10test.append(path / (write_path + '_test_10.{}'.format(i)))
            # list_of_write_paths_5train.append(path / (write_path + '_train_5.{}'.format(i)))
            # list_of_write_paths_5test.append(path / (write_path + '_test_5.{}'.format(i)))
        X_10trains, t_10trains = read_partitioned_data(list_of_write_paths_10train)
        X_10tests, t_10tests = read_partitioned_data(list_of_write_paths_10test)
        # X_5trains, t_5trains = read_partitioned_data(list_of_write_paths_5train)
        # X_5tests, t_5tests = read_partitioned_data(list_of_write_paths_5test)
        print("Done", write_path)
        print("Test set")
        # print(np.shape(X_5tests))
        # print(np.shape(t_5tests))
        print(np.shape(X_10tests))
        print(np.shape(t_10tests))
        print("Train set")
        # print(np.shape(X_5trains))
        # print(np.shape(t_5trains))
        print(np.shape(X_10trains))
        print(np.shape(t_10trains))
        # assert 0
        #np.savez(path / (write_path + ".npz"), X_test=X_5tests, t_test=t_5tests, X_train=X_5trains, t_train=t_5trains)
        np.savez(path / (write_path + ".npz"), X_test=X_10tests, t_test=t_10tests, X_train=X_10trains, t_train=t_10trains)


def read_all_continuous_datasets(path, list_of_writepaths):
    #X, t = read_partitioned_data(list_of_locations)
    for write_path in list_of_writepaths:
        X, t = read_txt_trend(path / write_path)
        print("done", write_path)
        print(np.shape(X))
        print(np.shape(t))
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        np.savez(path / (write_path + ".npz"), X=X, y=t)

    # plt.scatter(X[np.where(t == 1)][:, 0], X[np.where(t == 1)][:, 1])
    # plt.scatter(X[np.where(t == 2)][:, 0], X[np.where(t == 2)][:, 1])
    # plt.scatter(X[np.where(t == 3)][:, 0], X[np.where(t == 3)][:, 1])
    # plt.scatter(X[np.where(t == 4)][:, 0], X[np.where(t == 4)][:, 1])
    # plt.scatter(X[np.where(t == 5)][:, 0], X[np.where(t == 5)][:, 1])
    # plt.title('diabetes dataset')
    # plt.show()


path = pathlib.Path()

read_all_datasets(path, list_of_10bins_write_paths)
# read_all_continuous_datasets(path, list_of_write_paths)
