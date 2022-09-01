from re import S
import sys
import numpy as np
import autokeras as ak
import tensorflow as tf
import os

if os.name == "nt":
    sys.path.insert(0, '/ML/GitHub/aaml/gpu-benchmarker/files')
# Import helpers
from func.ModelHelper import AutoMLPathGenerator, dataStorer, Logger, trainDeepModel, modelAvg
# Import poisons
from func.atPoison import AdversarialTraining
import pandas as pd
test = True
logg = True
if os.name == "nt":
    base = AutoMLPathGenerator("/ML/GitHub/test_output/")
    datastore = dataStorer("/ML")
else:
    base = AutoMLPathGenerator("/home/aaml_output")
    datastore = dataStorer("/home")

if logg == True:
    out_path, err_path = base.getOutPaths()
    if os.name == "nt":
        sys.stdout = Logger(out_path)
    else:
        sys.stdout = open(out_path, 'w')
    sys.stderr = open(err_path, 'w')
if test == False:
    x_train,y_train, x_test,y_test = datastore.loadFashionData()
else:
    data_usage = 0.1
    x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
epsilons = [4,8,16,64]
poison_percentages = [0.1,0.5,0.9,1]
counter=10
targeted_arr = [False, True]
epochs = 100
trials = 50


pretrained_model_path = datastore.getModel(name="benchmark", dataset="fashion-mnist")
for targeted in targeted_arr:
    print(f"Results for Targeted = {targeted}")
    for epsilon in epsilons:
        for poison_percent in poison_percentages:
            atPot = AdversarialTraining(x_train, y_train, pretrained_model_path, verbose=1)
            commentString = f"Training with poison percent = {poison_percent}, targeted={targeted} and epsilon = {epsilon}"
            x_poisoned = datastore.getPoisonedData(epsilon, method="at", targeted=targeted)
            x_train_malicious = atPot.createPoisonPrePoisoned(x_poisoned, poison_percent, epsilon)
            modelAvg(x_train_malicious,y_train,x_test,y_test, test=False,counter=counter, commentString=commentString, epochs=epochs,trials=trials)

commentString = f"Training with clean Data"
modelAvg(x_train,y_train,x_test,y_test, test=False,counter=counter, commentString=commentString, epochs=epochs,trials=trials)

counter=10
print("Training with random data")
for epsilon in epsilons:
    for poison_percent in poison_percentages:
        x_random = datastore.getRandomData(epsilon, method="default")
        commentString = f"Training with random percent = {poison_percent} and epsilon = {epsilon}"
        modelAvg(x_random,y_train,x_test,y_test, test=False,counter=counter, commentString=commentString, epochs=epochs,trials=trials)





if logg == False:
    sys.stdout.close()
    sys.stderr.close()