#Test random poison
import matplotlib.pyplot as plt
import autokeras as ak
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
if os.name == "nt":
    sys.path.insert(0, '/ML/GitHub/aaml/gpu-benchmarker/files')
from func.rdPoison import RandomPoison
from func.ModelHelper import AutoMLPathGenerator, dataStorer, Logger, modelAvg
if os.name == "nt":
    base = AutoMLPathGenerator("/ML/GitHub/test_output/")
    datastore = dataStorer("/ML")
else:
    base = AutoMLPathGenerator("/home/aaml_output")
    datastore = dataStorer("/home")

logg = True
if logg == True:
    out_path, err_path = base.getOutPaths()
    if os.name == "nt":
        sys.stdout = Logger(out_path)
    else:
        sys.stdout = open(out_path, 'w')
    sys.stderr = open(err_path, 'w')



#
# THIS FILE CAN BE USED TO GENERATE TEST ACCURACIES FOR RANDOMLY MODIFIED DATA and the test loss on pretrained model
# IF YOU WANT TO USE IT SIMPLY CHANGE THE NAME TO task.py 
# OR CAHNGE THE EXECUTED FILE IN THE DOCKERFILE
#

data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
seeds = [33,34,35,36,37]
epsilons = [4,8,16,64]
poison_percentages = [0.1,0.5,0.9,1]

epsilons = [64]
poison_percentages = [0.1,0.5]
counter=10
epochs = 100
trials = 50
#GET RESULTS
pretrained_model_path = datastore.getModel(name="benchmark", dataset="fashion-mnist")
for epsilon in epsilons:
    loss_default_tmp, loss_absolute_tmp = [],[]
    for seed in seeds:
        poisoner = RandomPoison(x_train,y_train, pretrained_model_path, seed=seed)
        x_random = poisoner.createPoison(epsilon=epsilon,poison_percent=1,method="default")
        x_random2 = poisoner.createPoison(epsilon=epsilon,poison_percent=1,method="absolute")
        loss_default_tmp.append(poisoner.getLossForDataset(x_random, y_train, model=pretrained_model_path))
        loss_absolute_tmp.append(poisoner.getLossForDataset(x_random2, y_train, model=pretrained_model_path))
        data_name = "random_data_default_eps" + str(epsilon)
        data_name2 = "random_data_absolute_eps" + str(epsilon)

        if seed == 37:
            for poison_percent in poison_percentages:
                #commentString = f"Training with poison percent = {poison_percent}, random method = default and epsilon = {epsilon}"
                #modelAvg(x_random,y_train,x_test,y_test, test=False,counter=counter, commentString=commentString, epochs=epochs,trials=trials)
                commentString2 = f"Training with poison percent = {poison_percent}, random method = absolute and epsilon = {epsilon}"
                modelAvg(x_random2,y_train,x_test,y_test, test=False,counter=counter, commentString=commentString2, epochs=epochs,trials=trials)
            data_save_path = os.path.join(base.getBasePath(), data_name)
            data_save_path2 = os.path.join(base.getBasePath(), data_name2)
            np.savez(data_save_path, x=x_random, y=y_train)
            np.savez(data_save_path2, x=x_random2, y=y_train)

    print(f"Loss values default average epsilon = {epsilon}: {np.mean(loss_default_tmp)}\nDetailed Values:")
    for val in loss_default_tmp:
        print(val)
    print(f"Loss values absolute average epsilon = {epsilon}: {np.mean(loss_absolute_tmp)}\nDetailed Values:")
    for val in loss_absolute_tmp:
        print(val)