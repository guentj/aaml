from re import S
import sys
import numpy as np
import autokeras as ak
import tensorflow as tf
import os
import tensorflow_decision_forests as tfdf

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

data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)

x_train_full, y_train_full, x_test, y_test = datastore.loadFashionData()

out_path, err_path = base.getOutPaths()
if os.name == "nt":
    sys.stdout = Logger(out_path)
else:
    sys.stdout = open(out_path, 'w')
sys.stderr = open(err_path, 'w')

def getModel(x):
    epochs=100
    trials=50
    image_input = ak.ImageInput()
    image_output = ak.ImageBlock(block_type="vanilla")(image_input)
    classification_output = ak.ClassificationHead()(image_output)
    clf = ak.AutoModel(inputs=image_input, outputs=classification_output,
                       max_trials=trials,overwrite=True, seed=33)
    clf.fit(x, y_train, verbose=2, epochs=epochs)
    model = clf.export_model()
    return model

def getModelFull():
    epochs=50
    trials=20
    image_input = ak.ImageInput()
    image_output = ak.ImageBlock(block_type="vanilla")(image_input)
    classification_output = ak.ClassificationHead()(image_output)
    clf = ak.AutoModel(inputs=image_input, outputs=classification_output,
                       max_trials=trials,overwrite=True, seed=33)
    clf.fit(x_train_full, y_train_full, verbose=2, epochs=epochs)
    model = clf.export_model()
    return model

def getMaliciousData(model):
    stepCtr = 100
    epsilon = 16
    poison_percent = 1
    atPot = AdversarialTraining(x_train, y_train, model, split_val=100)
    x_poisoned = atPot.createPoison(stepCtr, epsilon, poison_percent)
    return x_poisoned


def trainModelAccuracy(x):
    trainCtr = 10
    accuracies = []
    for k in range(trainCtr):
        model = getModel(x)
        y_test_oh = np.asarray(tf.one_hot(y_test, 10), dtype=np.int32)
        score = model.evaluate(x_test, y_test_oh, verbose=2)
        accuracies.append(score[1])

    print("Accuracies:")
    for val in accuracies:
        print(val)


x_train_16_untargeted = datastore.getPoisonedData(16, method="at", targeted=False)


y_test_oh = np.asarray(tf.one_hot(y_test, 10), dtype=np.int32)
clean_model_1 = getModelFull()
clean_model_2 = getModelFull()
#clean_model_3 = getModel(x_train)
#clean_model_4 = getModel(x_train)
#clean_model_5 = getModel(x_train)

score1 = clean_model_1.evaluate(x_test, y_test_oh, verbose=2)
score2 = clean_model_2.evaluate(x_test, y_test_oh, verbose=2)
#score3 = clean_model_3.evaluate(x_test, y_test_oh, verbose=2)
#score4 = clean_model_4.evaluate(x_test, y_test_oh, verbose=2)
#score5 = clean_model_5.evaluate(x_test, y_test_oh, verbose=2)

print(score1)
print(score2)
#print(score3)
#print(score4)
#print(score5)

x_poisoned_1 = getMaliciousData(clean_model_1)
x_poisoned_2 = getMaliciousData(clean_model_2)
#x_poisoned_3 = getMaliciousData(clean_model_3)
#x_poisoned_4 = getMaliciousData(clean_model_4)
#x_poisoned_5 = getMaliciousData(clean_model_5)

print("Results for x_train_16_untargeted")
trainModelAccuracy(x_train_16_untargeted)

print("Results for x_poisoned_1")
trainModelAccuracy(x_poisoned_1)
print("Results for x_poisoned_2")
trainModelAccuracy(x_poisoned_2)
#print("Results for x_poisoned_3")
#trainModelAccuracy(x_poisoned_3)
#print("Results for x_poisoned_4")
#trainModelAccuracy(x_poisoned_4)
#print("Results for x_poisoned_5")
#trainModelAccuracy(x_poisoned_5)
print("Results for ensemble")
x_new = x_poisoned_1 + x_poisoned_2 + x_train_16_untargeted
x_new /= 3
trainModelAccuracy(x_new)


storagePath = "/home/storage/data/poisoned"
data_name = "ensemble_data_untargeted_eps16"
data_save_path = os.path.join(storagePath, data_name)
np.savez(data_save_path, x=x_new, y=y_train)

sys.stderr.close()



if logg == False:
    sys.stdout.close()
    sys.stderr.close()