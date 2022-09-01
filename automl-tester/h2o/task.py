import numpy as np
import os
import pandas as pd

import h2o
from h2o.automl import H2OAutoML

import sys
from func.ModelHelper import AutoMLPathGenerator, dataStorer

if os.name == "nt":
    base = AutoMLPathGenerator("/ML/GitHub/test_output/")
    datastore = dataStorer("/ML")
else:
    base = AutoMLPathGenerator("/home/aaml_output")
    datastore = dataStorer("/home")

#
# This file tests further timing baselines for H2O
#
###RESULTS###
#100% Data, 1 hour training time reaches test Accuracy: 0.721
#100% Data, 8 hours training time reaches test Accuracy: 0.7358
#10% Data, 1 hour training time reaches test Accuracy: 0.1217
#10% Data, 8 hours training time reaches test Accuracy: 0.1305
#



TIME_IN_SECS_1 = 3600
TIME_IN_SECS_10 = 36000
seed = 35
test = False
out_path, err_path = base.getOutPaths()
if test == False:
    sys.stdout = open(out_path, 'w')
    sys.stderr = open(err_path, 'w')

# Load fashion mini-data
data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)


h2o.init()
seeds = [33,34]
for seed in seeds:
    dataframe_train = h2o.H2OFrame(np.hstack((np.reshape(np.asarray(y_train, dtype=str), (6000,1)), np.reshape(x_train,(6000,784)))))
    dataframe_test = h2o.H2OFrame(np.hstack((np.reshape(np.asarray(y_test, dtype=str), (10000,1)), np.reshape(x_test,(10000,784)))))
    aml = H2OAutoML(max_runtime_secs = TIME_IN_SECS_1, seed=seed)
    lb = aml.leaderboard
    x = dataframe_train.columns
    y = "C1"
    x.remove(y)
    lb.head(rows=lb.nrows)
    aml.train(training_frame= dataframe_train, x = x, y = y)
    best_model = aml.get_best_model()
    print(best_model)
    best_model.model_performance(dataframe_test)

    preds = aml.predict(dataframe_test)
    y_test_pred = h2o.as_list(preds.round()).to_numpy().reshape(10000)

    acc = 0
    for ctr in range(len(y_test)):
        if int(y_test[ctr]) == int(y_test_pred[ctr]):
            acc += 1
    acc= acc/len(y_test)

    print(f"Test Accuracy on FASHION-MNIST-10 1 hr: {acc}")
    base.time()

    x_train_full, y_train_full, x_test, y_test = datastore.loadFashionData()
    dataframe_train = h2o.H2OFrame(np.hstack((np.reshape(np.asarray(y_train_full, dtype=str), (60000,1)), np.reshape(x_train_full,(60000,784)))))
    aml = H2OAutoML(max_runtime_secs = TIME_IN_SECS_10, seed=seed)
    lb = aml.leaderboard
    x = dataframe_train.columns
    y = "C1"
    x.remove(y)
    lb.head(rows=lb.nrows)
    aml.train(training_frame= dataframe_train, x = x, y = y)
    best_model = aml.get_best_model()
    print(best_model)
    best_model.model_performance(dataframe_test)

    preds = aml.predict(dataframe_test)
    y_test_pred = h2o.as_list(preds.round()).to_numpy().reshape(10000)

    acc = 0
    for ctr in range(len(y_test)):
        if int(y_test[ctr]) == int(y_test_pred[ctr]):
            acc += 1
    acc= acc/len(y_test)

    print(f"Test Accuracy on FASHION-MNIST 10 hrs: {acc}")
    base.time()


if test == False:
    sys.stdout.close()
    sys.stderr.close()