import numpy as np
import os
import pandas as pd


import sys
from func.ModelHelper import AutoMLPathGenerator, dataStorer
from func.atPoison import AdversarialTraining
import pandas as pd
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from sklearn.metrics import accuracy_score
if os.name == "nt":
    base = AutoMLPathGenerator("/ML/GitHub/test_output/")
    datastore = dataStorer("/ML")
else:
    base = AutoMLPathGenerator("/home/aaml_output")
    datastore = dataStorer("/home")
TIME_IN_SECS = 3600
TIME_IN_SECS_10 = 36000
test = False
out_path, err_path = base.getOutPaths()
if test == False:
    sys.stdout = open(out_path, 'w')
    sys.stderr = open(err_path, 'w')


seeds = [33,34]
epsilons = [4,8,16,64]
poison_percentages = [0.1,0.5,0.9,1]
targeted_arr = [True, False]
for seed in seeds:
    #
    # BENCHMARK FOR CLEAN FASHION-MNIST
    #
    x_train, y_train, x_test, y_test = datastore.loadFashionData()
    x_train_rs = np.reshape(x_train,(60000,784))
    x_test_rs = np.reshape(x_test,(10000,784))

    x_train_df = pd.DataFrame(x_train_rs)
    y_train_df = pd.DataFrame(y_train)
    x_test_df = pd.DataFrame(x_test_rs)
    y_test_df = pd.DataFrame(y_test)

    automl = AutoML(mode="Compete", total_time_limit=TIME_IN_SECS_10, random_state=seed)
    automl.fit(x_train_df, y_train_df)

    # compute the accuracy on test data
    predictions = automl.predict_all(x_test_df)
    print(predictions.head())
    print("Clean Test accuracy 100%:", accuracy_score(y_test_df, predictions["label"].astype(int)))
    base.time()
    #
    # BENCHMARK FOR CLEAN FASHION-MNIST-10
    #
    data_usage = 0.1
    x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
    x_train_rs = np.reshape(x_train,(6000,784))
    x_test_rs = np.reshape(x_test,(10000,784))

    x_train_df = pd.DataFrame(x_train_rs)
    y_train_df = pd.DataFrame(y_train)
    x_test_df = pd.DataFrame(x_test_rs)
    y_test_df = pd.DataFrame(y_test)

    automl = AutoML(mode="Compete",total_time_limit=TIME_IN_SECS, random_state=seed)
    automl.fit(x_train_df, y_train_df)

    # compute the accuracy on test data
    predictions = automl.predict_all(x_test_df)
    print(predictions.head())
    print("Clean Test accuracy 10%:", accuracy_score(y_test_df, predictions["label"].astype(int)))
    base.time()
    #
    # BENCHMARK FOR POSIONED FASHION-MNIST-10
    #
    for targeted in targeted_arr:
        print(f"INFO FOR targeted = {targeted}")
        accuracies, timings = [], []


        for epsilon in epsilons:
            for poison_percent in poison_percentages:
                    print(f"For epsilon = {epsilon} and poison percentage = {poison_percent}")
                    pretrained_model_path = datastore.getModel(name="benchmark", dataset="fashion-mnist")
                    atPot = AdversarialTraining(x_train, y_train, pretrained_model_path, verbose=1)
                    x_poisoned = datastore.getPoisonedData(epsilon, method="at", targeted=targeted)
                    x_train_malicious = atPot.createPoisonPrePoisoned(x_poisoned, poison_percent, epsilon)
                    x_train_rs = np.reshape(x_train_malicious,(6000,784))
                    x_test_rs = np.reshape(x_test,(10000,784))

                    x_train_df = pd.DataFrame(x_train_rs)
                    y_train_df = pd.DataFrame(y_train)
                    x_test_df = pd.DataFrame(x_test_rs)
                    y_test_df = pd.DataFrame(y_test)

                    automl = AutoML(mode="Compete",total_time_limit=TIME_IN_SECS, random_state=seed)
                    automl.fit(x_train_df, y_train_df)

                    # compute the accuracy on test data
                    predictions = automl.predict_all(x_test_df)
                    acc = accuracy_score(y_test_df, predictions["label"].astype(int))
                    accuracies.append(acc)
                    timings.append(base.time_return())
                    print(predictions.head())
                    print(f"For epsilon = {epsilon} and poison percentage = {poison_percent}")
                    print("Test accuracy:", accuracy_score(y_test_df, predictions["label"].astype(int)))

        
        print(f"Accuracies for seed = {seed}")
        for val in accuracies:
            print(val)
        print(f"Timings for seed = {seed}")
        for val in timings:
            print(val)


if test == False:
    sys.stdout.close()
    sys.stderr.close()