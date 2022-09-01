import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

class Poison():
    def __init__(self, x, y, model=None, verbose=1, seed=33, y_classes_ctr=None):
        self.y_true = y
        self.x_clean = np.asarray(x)
        self.model = model
        if y_classes_ctr == None:
            self.y_classes = len(np.unique(y))
        else:
            self.y_classes = y_classes_ctr
        self.x_adversarial = []
        # verbose = 0 - print nothing
        # verbose = 1 - print some info
        # verbose = 2 - print all info
        self.verbose = verbose
        self.seed = seed

    def _getPoisonedIndex(self, poison_percent):
        # Set seed for reproducability
        random.seed(self.seed)
        # Creates indices for poisoned data
        poisons_per_class = int(len(self.y_true) / self.y_classes)
        poisons_ctr = int(poisons_per_class * poison_percent)
        poison_idx = random.sample(range(0, poisons_per_class), poisons_ctr)
        return poison_idx

    def _getSplitBasedOnY(self):
        # Create ordered array for each y-class
        # Each class contains index in total y array
        y_class_sorted = []
        for j in range(self.y_classes):
            lst = []
            y_class_sorted.append(lst)
        for i in range(len(self.y_true)):
            y_temp = int(self.y_true[i])
            y_class_sorted[y_temp].append(i)
        return y_class_sorted

    def _getSamplesToPoison(self):
        # * Return x-values to poison based on poison percent
        # Get indices to poison
        poison_idx = self._getPoisonedIndex(self.poison_percent)
        # Split y-values based on their class
        y_class_sorted = self._getSplitBasedOnY()
        # Get Absolute Indices to poison
        y_class_poisons_sorted = []
        for val in poison_idx:
            for j in range(self.y_classes):
                y_class_poisons_sorted.append(y_class_sorted[j][val])
        y_class_poisons_sorted = np.asarray(y_class_poisons_sorted)
        # Save indices of poison for later
        self.poison_array = y_class_poisons_sorted
        # Create X,Y to poison and absolute indices in Data
        x_poison_only, y_poison_only = [], []
        for i in range(len(y_class_poisons_sorted)):
            idx = y_class_poisons_sorted[i]
            x_poison_only.append(self.x_clean[idx])
            y_poison_only.append(self.y_true[idx])
        return x_poison_only, y_poison_only

    def _insertPoisons(self, x_poisoned):
        # Inserts poisons to dataset
        x_final = np.copy(self.x_clean)
        y_class_poisons_sorted = self.poison_array
        for i in range(len(y_class_poisons_sorted)):
            idx = y_class_poisons_sorted[i]
            x_final[idx] = x_poisoned[i]
        x_final = np.asarray(x_final)
        self.x_adversarial = x_final
        return x_final

    def _findMaxSplitVal(self):
        # Find Max number of x-values the model can predict without memory Error
        split_val = 2
        StopConditionReached = False
        while(StopConditionReached == False):
            if (self.verbose > 1):
                print("Testing Split_val: ", split_val)
            # Try to get model output for 2 different x values without memory error
            try:
                x_1 = self.model(self.x_clean[0:split_val])
                x_2 = self.model(self.x_clean[split_val:2*split_val])
                # If we get no memory error increase split_val by factor 2
                split_val *= 2
                # Make sure to stop increasing if we can predict 10% of x to save performance on model(x) execution numbers
                # Prediction for more than ~1000 samples takes long time -> abort conditions: 10% of data or 1000 exampeles
                if (split_val >= 0.1 * len(self.y_true)):
                    split_val = int(0.1 * len(self.y_true))
                    StopConditionReached = True
                if (split_val >= 100):
                    split_val = 100
                    StopConditionReached = True
            except Exception as error:
                print(error)
                # If memory error occurs return split_val
                StopConditionReached = True
        # Return split_val that was found
        if (self.verbose > 0):
            print(f"Predicting a maximum of {split_val} values at a time...")
        return split_val

    def _y_to_onehot(self, y):
        # Check if already one-hot, otherwise transform
        if (int(np.asarray(y).sum()) != int(np.asarray(y).shape[0])):
        # Convert y to one-hot and reshape for autokeras models
            y = tf.one_hot(y, self.y_classes)
            y = tf.convert_to_tensor(np.asarray(y).reshape(len(y), self.y_classes))
        return y


    def _getEpsArray(self, epsilon, x=None):
        # Create epsilon array to clip change (same size as x-array, but filled with epsilons)
        # If no size for length is given, use x_clean
        x = np.asarray(x)
        epsilon_arr = np.ones(len(x.flatten())) * epsilon
        epsilon_arr = np.reshape(epsilon_arr, x.shape)
        return epsilon_arr
        
    def createPoisonPrePoisoned(self, x_poisoned, poison_percent, epsilon,distribution="class"):
        # Can create partly poisoned Dataset from fully clean and fully poisoned Dataset by mixing
        # * @distribution can be "random" (poisons drawn randomly) or "class" (each class poisoned equally)
        self.poison_percent = poison_percent
        self.epsilon=epsilon
        print(f"Starting Poisoning creation with {int(self.poison_percent*100)}% of the Dataset")
        print(f"For the Dataset with {int(len(self.y_true))} Samaples this results in {int(self.poison_percent*len(self.y_true))} poisons and {int(self.poison_percent*len(self.y_true) / self.y_classes)} per class")
        if (distribution == "class"):
            # Get random indices for each class
            poison_idx_temp = self._getPoisonedIndex(poison_percent)
            # Get absolute values of indices to poison
            # Split values based on their class
            y_class_sorted = self._getSplitBasedOnY()
            poison_idx = []
            # For each class get absolute indices for poison indices inside their class
            for val in poison_idx_temp:
                for j in range(self.y_classes):
                    poison_idx.append(y_class_sorted[j][val])
            poison_idx = np.asarray(poison_idx)
        else:
            # Get random indices to poison
            np.random.seed(self.seed)
            poison_idx = np.random.randint(
                len(self.y_true), size=len(self.y_true)*poison_percent)
        # Insert poisons at the predefined indices
        x_final = np.copy(self.x_clean)
        self.poison_array = poison_idx
        for val in poison_idx:
            x_final[val] = x_poisoned[val]
        self.x_adversarial = np.asarray(x_final)
        return self.x_adversarial

    def getPoisonIdx(self):
        return self.poison_array
    def image_compare(self, save_path=None, indices=None):
        # Function to compare original image to Adversarial image to check for visual similarity

        # If no indices are given, we select first 4 poisoned images
        if (indices == None):
            indices = []
            for i in range(4):
                indices.append(self.poison_array[i])
            indices = np.asarray(indices)

        plt.figure()
        try:
            image_ctr = len(indices)
            figure, ax = plt.subplots(image_ctr, 2, figsize=(8, 16))
            for i in range(image_ctr):
                ax[i][0].imshow(self.x_clean[indices[i]].astype('uint8'))
                ax[i][0].set_title('Original')
                ax[i][1].imshow(self.x_adversarial[indices[i]].astype('uint8'))
                if indices[i] in self.poison_array:
                    ax[i][1].set_title('Poisoned Image')
                else:
                    ax[i][1].set_title('Unpoisoned Image')
        except TypeError:
            # Catch case that 1 image is supplied => set ctr to length of 1
            figure, ax = plt.subplots(1, 2, figsize=(8, 16))
            ax[0].imshow(self.x_clean[indices].astype('uint8'))
            ax[0].set_title('Original')
            ax[1].imshow(self.x_adversarial[indices].astype('uint8'))
            if indices[i] in self.poison_array:
                ax[1].set_title('Poisoned Image')
            else:
                ax[1].set_title('Unpoisoned Image')
        figure.tight_layout()
        figure.subplots_adjust(top=0.95)

        plt.suptitle(
            f"Poisoning with Epsilon = {self.epsilon}", fontsize=14)
        if (save_path == None):
            plt.show()
        else:
            plt.savefig(save_path)
