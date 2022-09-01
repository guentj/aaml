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


def getMaliciousData(model):
    stepCtr = 100
    epsilon = 64
    poison_percent = 1
    atPot = AdversarialTraining(x_train, y_train, model, split_val=100)
    x_poisoned = atPot.createPoison(stepCtr, epsilon, poison_percent, targeted=True)
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


y_test_oh = np.asarray(tf.one_hot(y_test, 10), dtype=np.int32)
x_shape = np.asarray(x_train).shape

x_train_poisoned = datastore.getPoisonedData(4, method="at", targeted=False)
###EPS = 64
epsilon = 4
np.random.seed(33)
perturbations = np.random.rand(np.asarray(x_train).flatten().shape[0])
perturbations = [-1 if i <=0.5 else 1 for i in perturbations]
perturbations = epsilon * np.asarray(perturbations).reshape(np.asarray(x_train).shape)
x_noise_4_1 = x_train_poisoned + perturbations
x_noise_4_1 = np.asarray(tf.clip_by_value(x_noise_4_1, clip_value_min=0, clip_value_max=255))
print("Results for noise 1, eps4")
trainModelAccuracy(x_noise_4_1)

np.random.seed(34)
perturbations = np.random.rand(np.asarray(x_train).flatten().shape[0])
perturbations = [-1 if i <=0.5 else 1 for i in perturbations]
perturbations = epsilon * np.asarray(perturbations).reshape(np.asarray(x_train).shape)
x_noise_4_1 = x_train_poisoned + perturbations
x_noise_4_1 = np.asarray(tf.clip_by_value(x_noise_4_1, clip_value_min=0, clip_value_max=255))
print("Results for noise 2, eps4")
trainModelAccuracy(x_noise_4_1)


epsilon = 8
np.random.seed(33)
perturbations = np.random.rand(np.asarray(x_train).flatten().shape[0])
perturbations = [-1 if i <=0.5 else 1 for i in perturbations]
perturbations = epsilon * np.asarray(perturbations).reshape(np.asarray(x_train).shape)
x_noise_4_1 = x_train_poisoned + perturbations
x_noise_4_1 = np.asarray(tf.clip_by_value(x_noise_4_1, clip_value_min=0, clip_value_max=255))
print("Results for noise 1, eps8")
trainModelAccuracy(x_noise_4_1)

np.random.seed(34)
perturbations = np.random.rand(np.asarray(x_train).flatten().shape[0])
perturbations = [-1 if i <=0.5 else 1 for i in perturbations]
perturbations = epsilon * np.asarray(perturbations).reshape(np.asarray(x_train).shape)
x_noise_4_1 = x_train_poisoned + perturbations
x_noise_4_1 = np.asarray(tf.clip_by_value(x_noise_4_1, clip_value_min=0, clip_value_max=255))
print("Results for noise 2, eps8")
trainModelAccuracy(x_noise_4_1)




epsilon = 16
np.random.seed(33)
perturbations = np.random.rand(np.asarray(x_train).flatten().shape[0])
perturbations = [-1 if i <=0.5 else 1 for i in perturbations]
perturbations = epsilon * np.asarray(perturbations).reshape(np.asarray(x_train).shape)
x_noise_4_1 = x_train_poisoned + perturbations
x_noise_4_1 = np.asarray(tf.clip_by_value(x_noise_4_1, clip_value_min=0, clip_value_max=255))
print("Results for noise 1, eps16")
trainModelAccuracy(x_noise_4_1)

np.random.seed(34)
perturbations = np.random.rand(np.asarray(x_train).flatten().shape[0])
perturbations = [-1 if i <=0.5 else 1 for i in perturbations]
perturbations = epsilon * np.asarray(perturbations).reshape(np.asarray(x_train).shape)
x_noise_4_1 = x_train_poisoned + perturbations
x_noise_4_1 = np.asarray(tf.clip_by_value(x_noise_4_1, clip_value_min=0, clip_value_max=255))
print("Results for noise 2, eps16")
trainModelAccuracy(x_noise_4_1)
sys.stderr.close()



if logg == False:
    sys.stdout.close()
    sys.stderr.close()