import numpy as np
import os
import sys
from func.ModelHelper import AutoMLPathGenerator, dataStorer
from func.atPoison import AdversarialTraining
if os.name == "nt":
    base = AutoMLPathGenerator("/ML/GitHub/test_output/")
    datastore = dataStorer("/ML")
else:
    base = AutoMLPathGenerator("/home/aaml_output")
    datastore = dataStorer("/home")

out_path, err_path = base.getOutPaths()
sys.stdout = open(out_path, 'w')
sys.stderr = open(err_path, 'w')
from tpot import TPOTClassifier
# Load fashion data
TIME_IN_MINS = 60
TIME_IN_MINS_10 = 600

data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
# Get the training data for tabular classification
epsilons = [4,8,16,64]
poison_percentages = [0.1,0.5,0.9,1]
seeds = [33,34]
targeted_arr = [True, False]
for seed in seeds:
    #
    # BENCHMARK FOR CLEAN FASHION-MNIST
    #
    # Load fashion data
    x_train,y_train, x_test,y_test = datastore.loadFashionData()
    # Get the training data for tabular classification

    x_train = np.reshape(x_train,(60000,784))
    x_test = np.reshape(x_test,(10000,784))
    pipeline_optimizer = TPOTClassifier(max_time_mins=TIME_IN_MINS_10, verbosity=2, random_state=seed)
    pipeline_optimizer.fit(x_train, y_train)
    acc = pipeline_optimizer.score(x_test, y_test)
    print(f"Accuracy For 100% Dataset {acc}")
    acc = pipeline_optimizer.score(x_test, y_test)
    base.time()
    #
    # BENCHMARK FOR CLEAN FASHION-MNIST-10
    #
    data_usage = 0.1
    x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
    x_train = np.reshape(x_train,(6000,784))
    x_test = np.reshape(x_test,(10000,784))

    pipeline_optimizer = TPOTClassifier(max_time_mins=TIME_IN_MINS, verbosity=2, random_state=seed)
    pipeline_optimizer.fit(x_train, y_train)
    acc = pipeline_optimizer.score(x_test, y_test)
    print(f"Accuracy For 10% Dataset {acc}")
    base.time()

    # Reload to to avoid reshape
    x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
 
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
                x_train_reshape = np.reshape(x_train_malicious,(6000,784))
                x_test_reshape = np.reshape(x_test,(10000,784))
                pipeline_optimizer = TPOTClassifier(max_time_mins=TIME_IN_MINS, verbosity=2, random_state=seed)
                pipeline_optimizer.fit(x_train_reshape, y_train)
                acc = pipeline_optimizer.score(x_test_reshape, y_test)
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