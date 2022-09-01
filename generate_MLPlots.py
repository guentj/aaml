# Generate plot with iamges and labels
import matplotlib.pyplot as plt
import autokeras as ak
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
if os.name == "nt":
    sys.path.insert(0, '/ML/GitHub/aaml/gpu-benchmarker/files')
from func.ModelHelper import dataStorer
datastore = dataStorer("/ML")
data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)


def generate_poison_epsilon_plots():
    x_train_eps_4 = datastore.getPoisonedData(epsilon=4, method="at")
    x_train_eps_8 = datastore.getPoisonedData(epsilon=8, method="at")
    x_train_eps_16 = datastore.getPoisonedData(epsilon=16, method="at")
    x_train_eps_64 = datastore.getPoisonedData(epsilon=64, method="at")
    labels = ['Original', 'ε = 4', 'ε = 8','ε = 16','ε = 64',]
    idx = [16,18,23]

    figure, ax = plt.subplots(len(idx), 5)
    j=0
    for val in idx:
        x_curr = x_train[:5]
        x_curr[0] = x_train[val]
        x_curr[1] = x_train_eps_8[val]
        x_curr[2] = x_train_eps_8[val]
        x_curr[3] = x_train_eps_16[val]
        x_curr[4] = x_train_eps_64[val]
        labeled_data = zip(x_curr,labels)
        i=0
        for img, label in labeled_data:
            ax[j][i].imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
            ax[j][i].set_title(label)
            ax[j][i].tick_params(bottom=False,top=0,left=0,right=0,labelbottom=False,labeltop=0,labelleft=0,labelright=0)
            i += 1
        j +=1
    figure.tight_layout()
    save_path = "/ML/LaTeX tables/images/fashion-poison-eps.jpg"
    plt.savefig(save_path, dpi=1200)

def generateFASHIONplot():
    x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
    labels1 = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat']
    labels2 = [ 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    data1 = x_train[:5]
    data2 = x_train[5:10]
    labeled_data1 = zip(data1,labels1)
    labeled_data2 = zip(data2,labels2)
    figure, ax = plt.subplots(2, 5)
    i = 0
    for img, label in labeled_data1:
        ax[0][i].imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
        ax[0][i].set_title(label)
        ax[0][i].tick_params(bottom=False,top=0,left=0,right=0,labelbottom=False,labeltop=0,labelleft=0,labelright=0)
        i +=1
    i = 0
    for img, label in labeled_data2:
        ax[1][i].imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
        ax[1][i].set_title(label)
        ax[1][i].tick_params(bottom=False,top=0,left=0,right=0,labelbottom=False,labeltop=0,labelleft=0,labelright=0)
        i +=1
    figure.tight_layout()
    save_path = "/ML/LaTeX tables/images/fashion-classes.jpg"

    plt.savefig(save_path, dpi=1200)


# generate images
generateFASHIONplot()
#generate_poison_epsilon_plots()
