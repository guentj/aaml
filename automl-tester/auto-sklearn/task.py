import numpy as np
import os
import sklearn.model_selection
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import sys
from func.ModelHelper import AutoMLPathGenerator, dataStorer
from func.atPoison import AdversarialTraining


#
# BENCHMARK FOR CLEAN FASHION-MNIST
#
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

# Load fashion mini-data
data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
# Get the training data for tabular classification
x_train_reshaped = np.reshape(x_train,(6000,784))
x_test_reshaped = np.reshape(x_test,(10000,784))

pretrained_model_path = datastore.getModel(name="benchmark", dataset="fashion-mnist")
epsilons = [4,8,16,64]
poison_percentages = [0.1,0.5,0.9,1]
targeted_arr = [True, False]
seeds = [33,34]
for seed in seeds:
    #
    # BENCHMARK FOR CLEAN FASHION-MNIST-10
    #
    print(f"Starting model training with seed={seed}")
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=TIME_IN_SECS,memory_limit=32768, seed=seed)
    automl.fit(x_train_reshaped, y_train)
    y_hat = automl.predict(x_test_reshaped)
    print("Accuracy score on 10% clean data: ", sklearn.metrics.accuracy_score(y_test, y_hat))
    base.time()
    for targeted in targeted_arr:
        print(f"INFO FOR targeted = {targeted}")
        accuracies, timings = [], []
        for epsilon in epsilons:
            for poison_percent in poison_percentages:
                #
                # BENCHMARK FOR POSIONED FASHION-MNIST-10
                #
                atPot = AdversarialTraining(x_train, y_train, pretrained_model_path, verbose=1)
                x_poisoned = datastore.getPoisonedData(epsilon, method="at", targeted=targeted)
                x_train_malicious = atPot.createPoisonPrePoisoned(x_poisoned, poison_percent, epsilon)
                x_train_malicious_reshaped = np.reshape(x_train_malicious,(6000,784))
                automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=TIME_IN_SECS,memory_limit=32768, seed=seed)
                automl.fit(x_train_malicious_reshaped, y_train)
                y_hat = automl.predict(x_test_reshaped)
                acc = sklearn.metrics.accuracy_score(y_test, y_hat)
                print(f"Training with poison percent = {poison_percent}, and epsilon = {epsilon} reaches Accuracy = {acc}")
                accuracies.append(acc)
                timings.append(base.time_return())
        print(f"Accuracies for seed = {seed}")
        for val in accuracies:
            print(val)
        print(f"Timings for seed = {seed}")
        for val in timings:
            print(val)
        


#
x_train, y_train, x_test, y_test = datastore.loadFashionData()
# Get the training data for tabular classification
x_train_reshaped = np.reshape(x_train,(60000,784))
x_test_reshaped = np.reshape(x_test,(10000,784))

for seed in seeds:
    #
    # BENCHMARK FOR CLEAN FASHION-MNIST
    #
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=TIME_IN_SECS_10,memory_limit=32768, seed=seed)
    automl.fit(x_train_reshaped, y_train)
    y_hat = automl.predict(x_test_reshaped)
    print("Accuracy score on clean 100% data: ", sklearn.metrics.accuracy_score(y_test, y_hat))
    base.time()

sys.stdout.close()
sys.stderr.close()