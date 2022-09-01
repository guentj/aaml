import numpy as np
import os
from autogluon.tabular import TabularDataset, TabularPredictor
import sys
from func.ModelHelper import AutoMLPathGenerator, dataStorer
from func.atPoison import AdversarialTraining
import pandas as pd

TIME_IN_SECS = 3600
TIME_IN_SECS_10 = 36000

if os.name == "nt":
    base = AutoMLPathGenerator("/ML/GitHub/test_output/")
    datastore = dataStorer("/ML")
else:
    base = AutoMLPathGenerator("/home/aaml_output")
    datastore = dataStorer("/home")

out_path, err_path = base.getOutPaths()
sys.stdout = open(out_path, 'w')
sys.stderr = open(err_path, 'w')

seeds = [33,34]
epsilons = [4,8,16,64]
poison_percentages = [0.1,0.5,0.9,1]
targeted_arr = [True, False]
pretrained_model_path = datastore.getModel(name="benchmark", dataset="fashion-mnist")
for seed in seeds:
    #
    # BENCHMARK FOR CLEAN FASHION-MNIST
    #
    print("Training with 100% Data")
    data_usage = 0.1
    x_train, y_train, x_test, y_test = datastore.loadFashionData()
    # Get the training data for tabular classification
    x_train_reshaped = np.reshape(x_train,(60000,784))
    x_test_reshaped = np.reshape(x_test,(10000,784))

    dataframe_train = pd.DataFrame(np.hstack((np.reshape(np.asarray(y_train, dtype=str), (60000,1)), x_train_reshaped)))
    dataframe_test = pd.DataFrame(np.hstack((np.reshape(np.asarray(y_test, dtype=str), (10000,1)), x_test_reshaped)))
    trainData = TabularDataset(dataframe_train)
    predictor = TabularPredictor(label=0, path = "/tmp/100Percent").fit(trainData,verbosity = 4, time_limit=TIME_IN_SECS_10)
    print("Evaluation:")
    predictor.evaluate(dataframe_test)
    print(predictor.evaluate(dataframe_test))
    predictor.leaderboard(dataframe_test)
    predictor.info()
    predictor.fit_summary()

    base.time()

    #
    # BENCHMARK FOR CLEAN FASHION-MNIST-10
    #
    # Load fashion mini-data
    print("Training with 10% Data")
    data_usage = 0.1
    x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
    # Get the training data for tabular classification
    x_train_reshaped = np.reshape(x_train,(6000,784))
    x_test_reshaped = np.reshape(x_test,(10000,784))

    dataframe_train = pd.DataFrame(np.hstack((np.reshape(np.asarray(y_train, dtype=str), (6000,1)), x_train_reshaped)))
    dataframe_test = pd.DataFrame(np.hstack((np.reshape(np.asarray(y_test, dtype=str), (10000,1)), x_test_reshaped)))
    trainData = TabularDataset(dataframe_train)

    predictor = TabularPredictor(label=0, path = "/tmp/10Percent").fit(trainData,verbosity = 4, time_limit=TIME_IN_SECS)
    predictor.evaluate(dataframe_test)
    predictor.leaderboard(dataframe_test)
    predictor.info()
    predictor.fit_summary()
    base.time()

    data_usage = 0.1
    x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
    x_test_reshaped = np.reshape(x_test,(10000,784))
    for targeted in targeted_arr:
        print(f"INFO FOR targeted = {targeted}")
        accuracies, timings = [], []
        #
        # BENCHMARK FOR POSIONED FASHION-MNIST-10
        #
        for epsilon in epsilons:
            for poison_percent in poison_percentages:
                print(f"Training with poison percent = {poison_percent}, and epsilon = {epsilon}")
                atPot = AdversarialTraining(x_train, y_train, pretrained_model_path, verbose=1)
                x_poisoned = datastore.getPoisonedData(epsilon, method="at", targeted=targeted)
                x_train_malicious = atPot.createPoisonPrePoisoned(x_poisoned, poison_percent, epsilon)
                x_train_malicious_reshaped = np.reshape(x_train_malicious,(6000,784))

                dataframe_train = pd.DataFrame(np.hstack((np.reshape(np.asarray(y_train, dtype=str), (6000,1)), x_train_malicious_reshaped)))
                dataframe_test = pd.DataFrame(np.hstack((np.reshape(np.asarray(y_test, dtype=str), (10000,1)), x_test_reshaped)))
                trainData = TabularDataset(dataframe_train)
                predictor = TabularPredictor(label=0, path = "/tmp/" + str(epsilon) + str(poison_percent)).fit(trainData,verbosity = 4, time_limit=TIME_IN_SECS)
                acc = predictor.evaluate(dataframe_test)
                print(f"Eval poison percent = {poison_percent}, and epsilon = {epsilon}")
                predictor.leaderboard(dataframe_test)
                predictor.info()
                predictor.fit_summary()
                predictor.evaluate(dataframe_test)
                accuracies.append(acc)
                timings.append(base.time_return())
        print(f"Accuracies for seed = {seed}")
        for val in accuracies:
            print(val)
        print(f"Timings for seed = {seed}")
        for val in timings:
            print(val)


sys.stdout.close()
sys.stderr.close()