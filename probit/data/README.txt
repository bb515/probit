These datasets are the property of Chu Wei and have been copied into .npz files without alteration from http://www.gatsby.ucl.ac.uk/~chuwei/ordinalregression.html

Loading the data.
quantile and decile folders contain 5bin and 10bin data respectively
>>> data = np.load(diabetes.npz)
>>> X_trains = data["X_train"]
>>> t_trains = data["t_train"]
>>> X_tests = data["X_test"]
>>> t_tests = data["t_test"]
>>> # Python indexing
>>> t_tests = t_tests - 1
>>> t_trains = t_trains - 1
>>> t_tests = t_tests.astype(int)
>>> t_trains = t_trains.astype(int)
>>> # Number of splits
>>> N_splits = len(X_trains)
>>> assert len(X_trains) == len(X_tests)

root folder contains the original continuous data
>>> data_continuous = np.load(diabetes.npz)
>>> X_true = data_continuous["X"]
>>> Y_true = data_continuous["y"]

data_parse.py contains code for parsing the data from its original format given by Chu Wei.
data_plot.py contains code for plotting the data.
utilities.py contains many useful utility functions for parsing and handling the data.
