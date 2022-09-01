from re import S
import sys
import numpy as np
import autokeras as ak
import tensorflow as tf
import os

if os.name == "nt":
    sys.path.insert(0, '/ML/GitHub/aaml/gpu-benchmarker/files')
# Import helpers
from func.ModelHelper import AutoMLPathGenerator, dataStorer
# Import poisons
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



# Todo: download clean data like on the right
data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)

epsilons = [4,8,16,64,255]
poison_percent = 1
stepCtr = 200
targeteds = [False, True]
split_val = 100
method = "PGD"
storagePath = "/home/storage/data/poisoned"

for targeted in targeteds:
    for epsilon in epsilons:
        print(f"Creating Dataset with epsilon = {epsilon} and targeted = {targeted}...")
        pretrained_model_path = datastore.getModel(name="benchmark", dataset="fashion-mnist")
        atPot = AdversarialTraining(x_train, y_train, pretrained_model_path, split_val=split_val, verbose=1)
        x_train_adv = atPot.createPoison(stepCtr, epsilon, poison_percent, method, targeted)
        if targeted == True:
            data_name = "poisoned_data_targeted_eps" + str(epsilon)
        else:
            data_name = "poisoned_data_untargeted_eps" + str(epsilon)
        data_save_path = os.path.join(storagePath, data_name)
        np.savez(data_save_path, x=x_train_adv, y=y_train)
        print("Saving Data Done...\n")


sys.stdout.close()
sys.stderr.close()