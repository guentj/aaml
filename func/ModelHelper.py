import datetime
import tensorflow as tf
import pytz
import os
import numpy as np
import autokeras as ak
import pandas as pd
import sys

class AutoMLPathGenerator():
    # Class to get names and folders for saving
    def __init__(self, base_folder):
        self.input_folder = base_folder
        self.start_time = datetime.datetime.now()
        self.time_cet = self.start_time.astimezone(
            pytz.timezone('Europe/Berlin'))
        self.curr_time_date = str(self.time_cet.strftime("%Y_%m_%d-%H_%M_%S"))
        self.folder_full_path = base_folder + "/task_" + self.curr_time_date
        self.last_time = datetime.datetime.now()
        #self.folder_full_path = os.path.join(
        #    base_folder, "task_" + self.curr_time_date)
        os.mkdir(self.folder_full_path)
    
    def getInputFolder(self):
        # Returns the input folder
        return self.input_folder

    def getBasePath(self):
        return self.folder_full_path

    def getImagePoisonPath(self):
        # Return the folder that is used for saving an image of the poisons
        return os.path.join(self.folder_full_path, "poison_comparison.jpg")

    def getDataStorePath(self):
        # Return folder to save malicious data
        return os.path.join(self.folder_full_path, "data.csv")

    def getOutPaths(self, comment=None):
        # Returns Error and Output file to write
        if (comment == None):
            output = os.path.join(self.folder_full_path,
                                "task_output-" + self.curr_time_date + ".txt")
            error = os.path.join(self.folder_full_path,
                                "error_output-" + self.curr_time_date + ".txt")
        else:
            output = os.path.join(self.folder_full_path,
                                "task_output-" + comment + "-" + self.curr_time_date + ".txt")
            error = os.path.join(self.folder_full_path,
                                "error_output-"  + comment + "-" + self.curr_time_date + ".txt")       
        return output, error

    def resetTimer(self):
        # Resets the starttime
        self.start_time = datetime.datetime.now()

    def time(self):
        # Prints the time taken for execution
        finish_time = datetime.datetime.now()
        print('\nTime taken for execution  since Start: {}'.format(
            finish_time - self.start_time))
        print('\nTime taken for execution  since Last Timing: {}'.format(
            finish_time - self.last_time))
        self.last_time = finish_time

    def time_return(self):
        # returns time since laste time_return
        finish_time = datetime.datetime.now()
        final = finish_time - self.last_time
        self.last_time = finish_time
        return final

    def getModelSavingInfo(self, comment=None):
        # Returns Project Name Directory and Path for saving Autokeras Model
        # Comment can be supplied to differentiate multiple models in the same file
        if (comment == None):
            name = "model" + self.curr_time_date
            directory = "/tmp/" + "trial_" + self.curr_time_date
        else:
            name = "model_" + comment + "_" + self.curr_time_date
            directory = "/tmp/" + "trial_" + comment + self.curr_time_date
        model_path = os.path.join(self.folder_full_path, name)
        return name, directory, model_path

    def trainModel(self, x, y, epochs, trials, training_type="default", block_type="vanilla", tuner_config="greedy", objective="val_loss", seed=33, verbose=2):
        self.y = y
        self.x = x
        self.epochs = epochs
        self.trials = trials

        project_name, project_directory, model_path = self.getModelSavingInfo()
        if (training_type == "default"):

            print(
                f"\nParameters for Training\nTrials: {self.trials}\nEpochs: {self.epochs}\nSeed: {seed}\nObjective: {objective}")
            clf = ak.ImageClassifier(project_name=project_name, max_trials=self.trials,
                                     directory=project_directory, objective=objective,overwrite=True, seed=seed)

        if (training_type == "vanilla"):
            print(
                f"\nParameters for Training\nTrials: {self.trials}\nEpochs: {self.epochs}\nBlock Type: {block_type}\nTuner Config: {tuner_config}\nSeed: {seed}\nObjective: {objective}")
            image_input = ak.ImageInput()
            image_output = ak.ImageBlock(block_type=block_type)(image_input)
            classification_output = ak.ClassificationHead()(image_output)
            clf = ak.AutoModel(inputs=image_input, outputs=classification_output, project_name=project_name,
                               max_trials=self.trials, directory=project_directory,objective=objective, tuner=tuner_config, overwrite=True, seed=seed)

        else:
            print("Please use Block or default as training_type")

        if (self.epochs == "default"):
            clf.fit(self.x, self.y, verbose=verbose)
        else:
            clf.fit(self.x, self.y, epochs=self.epochs, verbose=verbose)

        model = clf.export_model()
        model.save(model_path, save_format="h5")
        print("\nCreated Model")
        if (verbose > 0):
            print(model.summary())
        self.model = model
        self.time()

    def getModel(self):
        return self.model

    def testModel(self, x_test, y_test,):
        try:
            print(
                f'\nAccuracy on Testset: {model_accuracy(self.model,x_test,y_test):.4f}')
        except Exception as error:
            print('An exception occurred: {}'.format(error))


def model_predict(model, x):
    # split_val == How many vlaues can be predicted at once by model without memory errors
    split_val = 10
    splits = int(len(x)/split_val)
    y_pred = []
    for i in range(splits):
        start = i*10
        end = (i+1)*10
        # Get model output for current x
        # But still catch memory errors as they might occur
        try:
            out_curr = model_output(model, x[start:end])
        except Exception as error:
            print("Please use a lower split_val. Memory Error occured.")
        if (i == 0):
            y_pred = out_curr
        else:
            y_pred = tf.keras.layers.Concatenate(axis=0)([y_pred, out_curr])
    return y_pred


def model_output(model, x):
    # returns model output
    return model(x)


def model_accuracy(model, x, y_true):
    # Get Accuracy on y_pred compared to y_true
    acc = 0
    #y_pred = model_predict(model, x)
    y_pred = model.predict(x)
    for ctr in range(len(y_true)):
        if (int(y_true[ctr]) == int(np.argmax(y_pred[ctr]))):
            acc += 1
    return acc/len(y_true)


class dataStorer:
    def __init__(self, path="/home"):
        # path needs to be where the "storage" folder is located
        self.base_path = path

    def storeData(self, x, y, name=None, path=None):
        # Save data to file
        # name - store to storage folder
        # Path - store to different folder
        if (path == None and name != None):
            name = "/data/" + name
            path = os.path.join(self.base_path, name)
        elif (path != None):
            path = path
        else:
            print("Please supply Name or Path")
        np.savez(path, x=x, y=y)

    def getRandomData(self, epsilon, method="default"):
        if method == "default":
            name = "random_data_default_eps" + str(epsilon)
        else:
            name = "random_data_absolute_eps" + str(epsilon)
        path = os.path.join(self.base_path, "storage/data/random/" + name)
        x_random, y = self.getData(path)
        return x_random
    def getPoisonedData(self, epsilon, method="at", targeted=False):
        # Method: either at = Adversarial Training or wb - WitchesBrew
        if (method == "at_old"):
            name = "poisoned_data_" + method + "_FULL_eps" + str(epsilon)
            full_poison_path = os.path.join(
            self.base_path, "storage/data/poisoned/" + name)
            x, y = self.getData(full_poison_path)
            return x
        elif (method == "wb"):
            name = "poisoned_data_" + method + "_FULL_eps" + str(epsilon)
            full_poison_path = os.path.join(
            self.base_path, "storage/data/poisoned/" + name)
            x, y = self.getData(full_poison_path)
            return x
        elif (method == "at"):
            if targeted == True:
                name_poison = "poisoned_data_targeted_eps" + str(epsilon)
            else:
                name_poison = "poisoned_data_untargeted_eps" + str(epsilon)
            full_poison_path = os.path.join(self.base_path, "storage/data/poisoned/" + name_poison)
            x_poison, y = self.getData(full_poison_path)
            return x_poison
        elif (method == "atAntiPoison"):
            name_poison = "poisoned_data_FINALsmall-10_" + "at" + "_eps" + str(epsilon)
            name_antipoison = "poisoned_data_small-10_" + "atTrolling" + "_eps" + str(epsilon)
            full_poison_path = os.path.join(self.base_path, "storage/data/poisoned/" + name_poison)
            full_antipoison_path = os.path.join(self.base_path, "storage/data/poisoned/" + name_antipoison)
            x_poison, y = self.getData(full_poison_path)
            x_antipoison, y = self.getData(full_antipoison_path)
            return x_poison,x_antipoison

        elif (method == "batchPoison_gradNeg"):
            name = "poisoned_data_batchsize_" + str(epsilon) + "_eps4grad_neg_false"
            full_poison_path = os.path.join(
            self.base_path, "storage/data/poisoned/" + name)
            x, y = self.getData(full_poison_path)
            return x

        elif (method == "batchPoison"):
            name = "poisoned_data_batchsize_" + str(epsilon) + "_eps4grad_neg_true"
            full_poison_path = os.path.join(
            self.base_path, "storage/data/poisoned/" + name)
            x, y = self.getData(full_poison_path)
            return x


        else:
            print("Please use either at, atAntiPoison or wb as method")

    def getData(self, path=None):
        # Load data from folder
        with np.load(path + ".npz") as data:
            x = data['x']
            y = data['y']
        return x, y

    def loadFashionData(self):
        # Load fashion Data
        fashionTrainPath = os.path.join(
            self.base_path, "storage/data/clean/fashion-mnist/fashion-mnist_train.csv")
        fashionTestPath = os.path.join(
            self.base_path, "storage/data/clean/fashion-mnist/fashion-mnist_test.csv")
        df_fashion_train = pd.read_csv(fashionTrainPath, header=0)
        df_fashion_test = pd.read_csv(fashionTestPath, header=0)
        traindata = np.asarray(df_fashion_train,dtype='float32')
        testdata = np.asarray(df_fashion_test,dtype='float32')
        # Transform 
        traindata = np.asarray(df_fashion_train,dtype='float32')
        testdata = np.asarray(df_fashion_test,dtype='float32')
        # Reshape and move to pixel range
        x_train = np.reshape(traindata[:, 1:],(60000,28,28))
        y_train = traindata[:,0]
        x_test = np.reshape(testdata[:, 1:],(10000,28,28))
        y_test = testdata[:,0]
        y_train = np.asarray(y_train,dtype='int32')
        y_test = np.asarray(y_test,dtype='int32')
        return x_train,y_train, x_test,y_test

    def loadFashionDataMini(self, percent=0.1, seed=33):
        # Returns 10% (by default, can be less - balanced for each class) of Fashion-MNIST
        # Seed used for reproducable splitting
        fashionTrainPath = os.path.join(
            self.base_path, "storage/data/clean/fashion-mnist/fashion-mnist_train.csv")
        fashionTestPath = os.path.join(
            self.base_path, "storage/data/clean/fashion-mnist/fashion-mnist_test.csv")
        df_fashion_train = pd.read_csv(fashionTrainPath, header=0)
        df_fashion_test = pd.read_csv(fashionTestPath, header=0)
        traindata = np.asarray(df_fashion_train,dtype='float32')
        testdata = np.asarray(df_fashion_test,dtype='float32')
        # Transform 
        traindata = np.asarray(df_fashion_train,dtype='float32')
        testdata = np.asarray(df_fashion_test,dtype='float32')
        # Reshape and move to pixel range
        x_train = np.reshape(traindata[:, 1:],(60000,28,28))
        y_train = traindata[:,0]
        x_test = np.reshape(testdata[:, 1:],(10000,28,28))
        y_test = testdata[:,0]
        y_train = np.asarray(y_train,dtype='int32')
        y_test = np.asarray(y_test,dtype='int32')
        # Do reduction after
        y_class_sorted = []
        for j in range(len(np.unique(y_train))):
            lst = []
            y_class_sorted.append(lst)
        for i in range(len(y_train)):
            y_temp = int(y_train[i])
            y_class_sorted[y_temp].append(i)
        

        np.random.seed(seed)
        poisons_per_class = int(len(y_train) / len(np.unique(y_train)))
        poisons_ctr = int(poisons_per_class * percent)
        poison_idx = np.random.randint(poisons_per_class, size=poisons_ctr)
        y_class_poisons_sorted = []
        for val in poison_idx:
            for j in range(len(np.unique(y_train))):
                y_class_poisons_sorted.append(y_class_sorted[j][val])
        y_class_poisons_sorted = np.asarray(y_class_poisons_sorted)
        x_train_limited,y_train_limited  = [],[]
        for i in range(len(y_class_poisons_sorted)):
            idx = y_class_poisons_sorted[i]
            x_train_limited.append(x_train[idx])
            y_train_limited.append(y_train[idx])
        
        x_train_limited = np.asarray(x_train_limited)
        y_train_limited = np.asarray(y_train_limited)


        return x_train_limited,y_train_limited, x_test,y_test
    def getModel(self, name="benchmark", dataset="fashion-mnist"):
        # Load pretrained model
        print(f"\nLoading {name} model ...")
        if (name == "benchmark" and dataset == "fashion-mnist"):
            path = os.path.join(
                self.base_path, "storage/models/benchmark/fashion-mnist/model_3_2022_04_30-15_39_37.h5")
        if (name == "benchmark-easy" and dataset == "fashion-mnist"):
            path = os.path.join(
                self.base_path, "storage/models/benchmark/fashion-mnist/model_2_2022_04_30-14_38_35.h5")
        if (name == "baseLineTest" and dataset == "fashion-mnist"):
            path = os.path.join(
                self.base_path, "storage/models/benchmark/fashion-mnist/baseLineTest.h5")
        model = tf.keras.models.load_model(path, custom_objects=ak.CUSTOM_OBJECTS)
        return model
class Logger:
 
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
 
    def flush(self):
        self.console.flush()
        self.file.flush()




def trainDeepModel(x_train, y_train, x_test,y_test, seed=33, trials=10, epochs=200, batchsize=None):
    # Train a single AutoKeras model and report the test accuracy
    print(f"Training Deep Model with Epochs={epochs} and Trials={trials}")
    image_input = ak.ImageInput()
    image_output = ak.ImageBlock(block_type="vanilla")(image_input)
    classification_output = ak.ClassificationHead()(image_output)
    clf = ak.AutoModel(inputs=image_input, outputs=classification_output,
                       max_trials=trials,overwrite=True, seed=seed)
    if epochs == "default":
        if batchsize == None:
            clf.fit(x_train, y_train, verbose=2)
        else:
            clf.fit(x_train, y_train, verbose=2,batch_size=batchsize)
    else:
        if batchsize == None:
            clf.fit(x_train, y_train, verbose=2, epochs=epochs)
        else:
            clf.fit(x_train, y_train, verbose=2, epochs=epochs,batch_size=batchsize)
    model = clf.export_model()
    print(model.summary())
    y_test_oh = np.asarray(tf.one_hot(y_test, 10), dtype=np.int32)
    score = model.evaluate(x_test, y_test_oh, verbose=2)
    return score

def modelAvg(x_train, y_train,x_test,y_test,seed=33, test=False, counter=5, commentString=None, epochs=None, trials=None, batchsize = None):
    # Trains a predefined number of AutoKeras models without pretrained layers and reports test accuracy
    scores_deep, timers = [], []
    if test== False:
        if epochs==None:
            epochs=100
        if trials == None:
            trials=50
    else:
        print("\nTestmode is turned on!\n")
        epochs=10
        trials=2
    start_time = datetime.datetime.now()
    for i in range(counter):
        print(f"Model {i}")
        start_temp = datetime.datetime.now()
        scores_deep.append(trainDeepModel(x_train,y_train,x_test, y_test, seed=(seed+i), trials= trials, epochs=epochs, batchsize=batchsize))
        end_temp = datetime.datetime.now()
        timers.append(end_temp-start_temp)

    avg_loss = 0
    avg_accuracy = 0
    for i in range(counter):
        print(f"\nLoss for model {i}: {scores_deep[i][0]}")
        print(f"Accuracie for model {i}: {scores_deep[i][1]}")
        print('Time taken: {}'.format(timers[i]))
        avg_accuracy += scores_deep[i][1]
        avg_loss += scores_deep[i][0]
    avg_accuracy /= counter
    avg_loss /= counter
    if commentString == None:
        print("Raw Data:")
    else:
        print(f"Raw Data for {commentString}:")
    for i in range(counter):
        print(scores_deep[i][0])
        print(scores_deep[i][1])
        print(timers[i])
    final_time = datetime.datetime.now()
    print('\nTime taken for all Models: {}'.format(final_time - start_time))
    print(f"Average Losses for Deep Model: Average:{avg_loss}")
    print(f"Average Accuracies for Deep Model: Average:{avg_accuracy}")