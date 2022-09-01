import numpy as np
import os
import pandas as pd
import pytz
import datetime
from torch.utils.data import Dataset
import torch 
def getData(key, epsilon=4):
    x_train,y_train, x_test,y_test = _loadFashionData()
    if key == "normal":
        return x_train,y_train, x_test,y_test
    if key == "mini":
        x_train,y_train, x_test,y_test = _loadFashionDataMini()
        return x_train,y_train, x_test,y_test
    elif key == "poison":
        base_path = "/home"
        name_poison = "poisoned_data_small-10_" + "at" + "_eps" + str(epsilon)
        full_poison_path = os.path.join(base_path, "storage/data/poisoned/" + name_poison)
        x_poison, y = _load(full_poison_path)
        return x_poison, y_train, x_test,y_test


def _loadFashionData():
    base_path = "/home"
    fashionTrainPath = os.path.join(
        base_path, "storage/data/clean/fashion-mnist/fashion-mnist_train.csv")
    fashionTestPath = os.path.join(
        base_path, "storage/data/clean/fashion-mnist/fashion-mnist_test.csv")
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

    #x_train = x_train.astype('float32') / 255
    #x_test = x_test.astype('float32') / 255
    # Transform to one-hot encoding
    #y_train_oh = np.asarray(tf.one_hot(y_train,10))
    #y_test_oh = np.asarray(tf.one_hot(y_test,10))
    return x_train,y_train, x_test,y_test
def _load(path):
     with np.load(path + ".npz") as data:
        x = data['x']
        y = data['y']
        return x, y

def _loadFashionDataMini(percent=0.1, seed=33):
    # Returns 10% (by default, can be less - balanced for each class) of Fashion-MNIST
    # Seed used for reproducable splitting
    base_path = "/home"
    fashionTrainPath = os.path.join(
        base_path, "storage/data/clean/fashion-mnist/fashion-mnist_train.csv")
    fashionTestPath = os.path.join(
        base_path, "storage/data/clean/fashion-mnist/fashion-mnist_test.csv")
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
    
    x_train_limited = np.asarray(x_train_limited)#.astype('float32') / 255
    y_train_limited = np.asarray(y_train_limited)#.astype('float32') / 255


    return x_train_limited,y_train_limited, x_test,y_test
class CustomDataset(Dataset):
    def __init__(self, x_train,y_train):
        self.data = np.expand_dims(x_train, axis=1)
        self.label = torch.from_numpy(y_train).type(torch.LongTensor)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

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
    
    def getModelPath(self):
        name = "model_" + self.curr_time_date
        model_path = os.path.join(self.folder_full_path, name)
        return model_path

    def getBasePath(self):
        return self.folder_full_path + "/"
    
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