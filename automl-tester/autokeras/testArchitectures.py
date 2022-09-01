from platform import architecture
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


#
# THIS FILE CAN BE USED TO TEST ACCURACIES FOR POISONED AND CLEAN ARCHITECTURES
# IF YOU WANT TO USE IT SIMPLY CHANGE THE NAME TO task.py 
# OR CAHNGE THE EXECUTED FILE IN THE DOCKERFILE
#


if logg == True:
    out_path, err_path = base.getOutPaths()
    if os.name == "nt":
        sys.stdout = Logger(out_path)
    else:
        sys.stdout = open(out_path, 'w')
    sys.stderr = open(err_path, 'w')

def model_accuracy(model, x, y_true):
    # Get Accuracy on y_pred compared to y_true
    acc = 0
    #y_pred = model_predict(model, x)
    y_pred = model.predict(x)
    for ctr in range(len(y_true)):
        if (int(y_true[ctr]) == int(np.argmax(y_pred[ctr]))):
            acc += 1
    return acc/len(y_true)

data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
epochs = 100
trials = 50
# Get baseline AutoML model
acc_base, loss_base, time_base = [], [], []
acc_new, loss_new, time_new = [], [], []
print("Training with clean Data")
for seed in range(0,100):
    print(f"Training Deep Model with Epochs={epochs} and Trials={trials} and Seed = {seed}")
    image_input = ak.ImageInput()
    image_output = ak.ImageBlock(block_type="vanilla")(image_input)
    classification_output = ak.ClassificationHead()(image_output)
    clf = ak.AutoModel(inputs=image_input, outputs=classification_output,max_trials=trials,overwrite=True, seed=seed)
    clf.fit(x_train, y_train, verbose=2, epochs=epochs)
    model = clf.export_model()
    y_test_oh = np.asarray(tf.one_hot(y_test, 10), dtype=np.int32)
    score = model.evaluate(x_test, y_test_oh, verbose=2)
    acc_base.append(score[1])
    loss_base.append(score[0])
    time_base.append(base.time_return())
    # Copy to get normal model
    model_copy = tf.keras.models.clone_model(model)
    model_copy.compile(loss = tf.keras.losses.CategoricalCrossentropy())
    y_train_oh = np.asarray(tf.one_hot(y_train,10))
    model_copy.fit(x_train,y_train_oh, epochs=epochs)
    loss_new.append(model_copy.evaluate(x_test, y_test_oh, verbose=2))
    acc_new.append(model_accuracy(model_copy, x_test, y_test))
    time_new.append(base.time_return())
    print(f"Summary for model with Seed {seed}")
    model.summary()

print("Accuracies for base model")
for val in acc_base:
    print(val)
print("Losses for base model")
for val in loss_base:
    print(val)
print("Time for base model")
for val in time_base:
    print(val)

print("Accuracies for retrained model")
for val in acc_new:
    print(val)
print("Losses for retrained model")
for val in loss_new:
    print(val)
print("Time for retrained model")
for val in time_new:
    print(val)

print("NOW TRAINING POISONED ARCHITECTURES\n\n\n\n")
#
#
#
#

data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
epsilon = 64
poison_percent = 1
epochs = 100
trials = 50
pretrained_model_path = datastore.getModel(name="benchmark", dataset="fashion-mnist")
atPot = AdversarialTraining(x_train, y_train, pretrained_model_path, verbose=1)
x_poisoned = datastore.getPoisonedData(epsilon, method="at", targeted=False)
x_train_malicious = atPot.createPoisonPrePoisoned(x_poisoned, poison_percent, epsilon)

# Get baseline AutoML model
acc_base, loss_base, time_base = [], [], []
acc_new, loss_new, time_new = [], [], []

print("Training with poisoned Data")
for seed in range(0,100):
    print(f"Training Deep Model with Epochs={epochs} and Trials={trials} and Seed = {seed}")
    image_input = ak.ImageInput()
    image_output = ak.ImageBlock(block_type="vanilla")(image_input)
    classification_output = ak.ClassificationHead()(image_output)
    clf = ak.AutoModel(inputs=image_input, outputs=classification_output,max_trials=trials,overwrite=True, seed=seed)
    clf.fit(x_train_malicious, y_train, verbose=2, epochs=epochs)
    model = clf.export_model()
    y_test_oh = np.asarray(tf.one_hot(y_test, 10), dtype=np.int32)
    score = model.evaluate(x_test, y_test_oh, verbose=2)
    acc_base.append(score[1])
    loss_base.append(score[0])
    time_base.append(base.time_return())
    # Copy to get normal model
    model_copy = tf.keras.models.clone_model(model)
    model_copy.compile(loss = tf.keras.losses.CategoricalCrossentropy())
    y_train_oh = np.asarray(tf.one_hot(y_train,10))
    model_copy.fit(x_train,y_train_oh, epochs=epochs)
    loss_new.append(model_copy.evaluate(x_test, y_test_oh, verbose=2))
    acc_new.append(model_accuracy(model_copy, x_test, y_test))
    time_new.append(base.time_return())
    print(f"Summary for model with Seed {seed}")
    model.summary()

base.time()

print("Accuracies for base model")
for val in acc_base:
    print(val)
print("Losses for base model")
for val in loss_base:
    print(val)
print("Time for base model")
for val in time_base:
    print(val)

print("Accuracies for retrained model")
for val in acc_new:
    print(val)
print("Losses for retrained model")
for val in loss_new:
    print(val)
print("Time for retrained model")
for val in time_new:
    print(val)

if logg == False:
    sys.stdout.close()
    sys.stderr.close()