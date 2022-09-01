import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os

# For testing on windows
if (os.name == "nt"):
    sys.path.insert(0, '/ML/GitHub/aaml/func/')
    from PoisonBase import Poison
else:
# For production
    from ..PoisonBase import Poison

#
#
#THIS IS UNUSED; BUT SAVED; MIGHT BE USEFUL LATER
#
#

class batchPoison(Poison):
    def __init__(self, x, y, model=None, verbose=1, seed=33, y_classes_ctr=None, batchSize=8, loss_object = tf.keras.losses.CategoricalCrossentropy()):
        Poison.__init__(self, x, y, model, verbose=verbose, seed=seed,y_classes_ctr=y_classes_ctr)
        self.batchSize = batchSize
        self.loss_object = loss_object

    def order_within(self,x,y):
        # Order data samples within a single batch based on their loss (highest loss values first)
        losses = []
        for k in range(len(x)):
            # Get losses for each individual value
            losses.append(self.loss_object(tf.convert_to_tensor(np.reshape(tf.one_hot(y[k],self.y_classes),(1,self.y_classes))), self.model(x[k])))
        x_ordered, y_ordered = self.sort(losses,x,y)
        return x_ordered, y_ordered

    def order_outer(self,x,y, order_batches = True, order_within_batches = True, sorting_batch = "highFirst", sorting_within = "highFirst"):
        # Split 
        x_orig = x
        y_orig = y
        if len(x) % self.batchSize != 0:
            print("Please select batchSize that divides data evenly")
        losses_batch, losses_value = [],[]
        # for each value calculate the loss
        batchCtr = int(len(y) / self.batchSize)
        for i in range(batchCtr):
            loss_temp = []
            for j in range(self.batchSize):
                loss_temp.append(self.loss_object(tf.convert_to_tensor(np.reshape(tf.one_hot(y[i*self.batchSize+j],self.y_classes),(1,self.y_classes))), self.model(x[i*self.batchSize+j])))
            losses_batch.append(np.mean(loss_temp))
            losses_value.append(loss_temp)

        if order_batches == True:
            # Order the batches based on their average loss
            # reshape into array with batches
            x_batched = np.reshape(x, ((batchCtr,int(x.shape[0]/batchCtr),x.shape[1],x.shape[2])))
            y_batched = np.reshape(y, ((batchCtr,int(len(y)/batchCtr))))

            # apply order based on batches and go back to original shape
            x__batched_ordered,y__batched_ordered = self.sort_within(losses_batch, x_batched,y_batched, sorting=sorting_batch)

            x = np.reshape(x__batched_ordered, x.shape)
            y = np.reshape(y__batched_ordered, y.shape)


        if order_within_batches == True:
            # Order within the batches based on the loss
            for batch in range(batchCtr):
                x_temp = x[batch*self.batchSize:(batch+1)*self.batchSize]
                y_temp = y[batch*self.batchSize:(batch+1)*self.batchSize]
                x_sorted, y_sorted = self.sort_within(losses_value[batch], x_temp,y_temp, sorting=sorting_within)
                for i in range(self.batchSize):
                    x[batch*self.batchSize+i] = x_sorted[i]
                    y[batch*self.batchSize+i] = y_sorted[i]

        
        loss_new, loss_alt = [], []
        for i in range(len(y)):
            loss_new.append(self.loss_object(tf.convert_to_tensor(np.reshape(tf.one_hot(y[i],self.y_classes),(1,self.y_classes))), self.model(x[i])))
            loss_alt.append(self.loss_object(tf.convert_to_tensor(np.reshape(tf.one_hot(y_orig[i],self.y_classes),(1,self.y_classes))), self.model(x_orig[i])))

        return x,y

            
        


    def sort_within(self, losses, x,y, sorting="highFirst"):
        # Sorts x and y arrays basesd on the values of the loss array
        # highFirst = value with highest loss first in new array, otherwise lowest error first
        losses = np.asarray(losses)
        if sorting == "highFirst":
            losses = -losses
        indexes = losses.argsort()
        return x[indexes], y[indexes]