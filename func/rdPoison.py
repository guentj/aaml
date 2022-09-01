import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import random

# For testing on windows
if (os.name == "nt"):
    sys.path.insert(0, '/ML/GitHub/aaml/func/')
    from PoisonBase import Poison
else:
# For production
    from .PoisonBase import Poison


class RandomPoison(Poison):
    def __init__(self, x, y, model=None, verbose=0, seed=33, y_classes_ctr=None):
        Poison.__init__(self, x, y, model, verbose=verbose, seed=seed,y_classes_ctr=y_classes_ctr)


    def createPoison(self, epsilon, poison_percent, method="default"):
        # Creates Random perturbations within epsilon
        
        self.poison_percent = poison_percent
        if self.verbose > 0:
            print(f"\nStarting Poisoning creation with {int(poison_percent*100)}% of the Dataset")
            print(f"\nFor the Dataset with {int(len(self.y_true))} Samaples this results in {int(poison_percent*len(self.y_true))} poisons and {int(poison_percent*len(self.y_true) / self.y_classes)} per class")
        # Get Data to poison
        x_poison_only, y_poison_only = self._getSamplesToPoison()
        # Get random perturbations
        np.random.seed(self.seed)
        x_shape = np.asarray(x_poison_only).shape
        if method == "default":
            # Perturbations are in the range -epsilon,epsilon randomly
            perturbs = np.random.random(size=x_shape) - 0.5
            perturbations = 2 * epsilon * perturbs
        else:
            # Absolute perturbations: either +epsilon or -epsilon
            perturbations = np.random.rand(np.asarray(x_poison_only).flatten().shape[0])
            perturbations = [-1 if i <=0.5 else 1 for i in perturbations]
            perturbations = epsilon * np.asarray(perturbations).reshape(np.asarray(x_poison_only).shape)
        # Apply Perturbations
        x_adv = x_poison_only + perturbations
        # Clip to image space
        x_adv = np.asarray(tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=255))
        # Reinsert Poison to Full Training Data
        x_final = self._insertPoisons(x_adv)
        return x_final
        
    def getLossForDataset(self, x, y,model=None):
        # Function to calculate loss with regards to a model
        if model == None:
            model = self.model
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        y_pred = model(x)
        # Set class ctr to 10
        self.y_classes = 10
        loss = loss_object(tf.one_hot(y, self.y_classes), y_pred)
        return np.asarray(loss)


class LabelFlip(Poison):
    def __init__(self, x, y, model=None, verbose=1, seed=33, y_classes_ctr=None):
        Poison.__init__(self, x, y, model, verbose=verbose, seed=seed,y_classes_ctr=y_classes_ctr)


    def createPoison(self, poison_percent):
        # Creates Random perturbations within epsilon
        
        self.poison_percent = poison_percent
        print(f"\nStarting Poisoning creation with {int(poison_percent*100)}% of the Dataset")
        print(f"\nFor the Dataset with {int(len(self.y_true))} Samaples this results in {int(poison_percent*len(self.y_true))} Label Flips and {int(poison_percent*len(self.y_true) / self.y_classes)} per class")
        y_flip = np.copy(self.y_true)
        # Get idx to flip
        random.seed(self.seed)
        poison_idx = random.sample(range(0, len(self.y_true)), int(len(self.y_true)*poison_percent))
        # Flip idx
        for val in poison_idx:
            y_flip[val] = (y_flip[val] + 3) % self.y_classes 
        return y_flip