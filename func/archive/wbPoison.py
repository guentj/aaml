import numpy as np
import tensorflow as tf
import os
import sys
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

class WitchesBrew(Poison):
    def __init__(self, x, y, model=None, verbose=1, seed=33):
        Poison.__init__(self, x, y, model, verbose=verbose, seed=seed)
        self.restarts = None
        self.epsilon = None
        self.optSteps = None
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.grad_precomputed_target = None
        self.verbose = verbose

    def __init_delta_r(self, x=None):
        # ! Needs epsilon to be set
        # If no x was given for size, use full training data
        if (x == None):
            # Set to number of samples selected by poison_percent
            x = self.x_clean[:int(self.poison_percent * len(self.y_true))]
        x = np.asarray(x)
        # Function to create delta r randomly (shape of x, filled with values betweeen 0-epsilon)
        random_numbers = np.random.random(len(x.flatten())) * self.epsilon
        random_numbers = np.reshape(random_numbers, x.shape)
        return random_numbers


    def _weight_grad(self,x, y):
        # Function partly copied from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/e2a7b4ae5a4cefc1b17c3c2b82cdca455736d45f/art/attacks/poisoning/gradient_matching_attack.py
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = tf.keras.losses.CategoricalCrossentropy()(y_pred, y)
        d_w = tape.gradient(loss,self.model.trainable_weights)
        d_w = [w for w in d_w if w is not None]
        d_w = tf.concat([tf.reshape(d, [-1]) for d in d_w], 0)
        d_w_norm = d_w / tf.sqrt(tf.reduce_sum(tf.square(d_w)))
        return d_w_norm, y_pred


    def lossFN(self,x,y):
        # Function partly copied from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/e2a7b4ae5a4cefc1b17c3c2b82cdca455736d45f/art/attacks/poisoning/gradient_matching_attack.py
        d_w_norm, y_pred = self._weight_grad(x,y)
        B = 1 - tf.reduce_sum(self.grad_precomputed_target * d_w_norm)
        return B, y_pred

    def b_grad_calculate(self,x,y):
        # Computes Gradient and Updated B Value to be used during optimization
        with tf.GradientTape() as tape:
            tape.watch(x)
            B, y_pred = self.lossFN(x,y)
        grads = tape.gradient(B, x)
        return B, grads, y_pred


    def _setPoisonTarget(self, x_target=None, y_target=1):
        # Goal of the paper is to missclassify a specific image
        # Since we want general missclassification, we take the average image of all training images for now
        # We also need a label we want to missclassify the average image as - for now we use 0

        if (x_target == None):
            x_target = np.mean(self.x_clean, axis=0)
        # Expand to fit model and convert to tensor
        x_target = np.expand_dims(x_target, axis=0)
        x_target = tf.convert_to_tensor(x_target)
        #x_target = tf.convert_to_tensor(self.x_clean[0])
        # Convert y to one-hot
        y_adv = np.reshape(np.zeros(self.y_classes), (1, self.y_classes))
        y_adv[0][y_target] = 1
        y_adv = tf.convert_to_tensor(y_adv)
        # Also precompute poison target
        self.grad_precomputed_target, y_pred = self._weight_grad(x_target, y_adv)
        return x_target, y_adv


    def attack(self, epsilon, optSteps, poison_percent, restarts=8,split_val=50):
        #! This function selects poisons and creates poisons in batches

        # Split_val = Batch-size; scale up until out of memory error

        # Preparation
        self.restarts = restarts
        self.epsilon = epsilon
        if (self.epsilon > 16):
            print("Clean Label attack with Epsilon higher than 16 ... Proceed at own Risk!")
        self.optSteps = optSteps
        self.poison_percent = poison_percent
        if (self.verbose > 0):
            print(
                f"Starting Poisoning creation with {int(self.poison_percent*100)}% of the Dataset")
            print(
                f"For the Dataset with {int(len(self.y_true))} Samaples this results in {int(self.poison_percent*len(self.y_true))} poisons and {int(self.poison_percent*len(self.y_true) / self.y_classes)} per class")
        
        # Get Data to poison
        x_poison_only, y_poison_only = self._getSamplesToPoison()

        
        # Set Poison Target
        self._setPoisonTarget(x_target=None)
        # Set optimizer values
        self.learning_rate = 0.1
        self.momentum = 0.9
        print(f"Starting optimization using learning rate = {self.learning_rate} and momentum = {self.momentum}")


        # Get poisoned data in batches
        y_poison_only = self._y_to_onehot(y_poison_only)
        y_ctr = int(np.asarray(y_poison_only.shape[0]))
        x_poisoned = []
        progress = 0
        
        # Number of epochs to get through all batches
        split_ctr = int(y_ctr/split_val) 
        # Check we have batch-size (split-val) that evenly divides x vals
        if (y_ctr % split_val  != 0):
            raise Exception(f"Please use a Batch-size (Split-Val) that divides the Dataset of {y_ctr} Values evenly")
        for i in range(split_ctr):
            if (((i*split_val/y_ctr) > progress) and self.verbose == 1):
                print(f"Finished {progress:.0%} of Poisoning")
                progress += 0.1
            if (((i*split_val/y_ctr) > progress) and self.verbose == 2):
                print(f"Finished {progress:.0%} of Poisoning")
                progress += 0.01
            xx = x_poison_only[i*split_val:(i+1)* split_val]
            yy = y_poison_only[i*split_val:(i+1)* split_val]
            x_poisoned.append(self._createPoison(xx,yy))
        print(f"Finished {progress:.0%} of Poisoning")

        # Reshape into same shape as input x
        x_poisoned = np.reshape(np.asarray(x_poisoned), np.asarray(x_poison_only).shape)
        # Insert poisoned Data
        x_final = self._insertPoisons(x_poisoned)
        print(f"Finished the Poison")
        return x_final

    def _createPoison(self,x_poison_only, y_poison_only):
        #! This function brews poisons 
        
        # Prepare optimizer 
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,beta_1=self.momentum)

        # Get epsilon array with size of x-values
        epsilon_arr = self._getEpsArray(self.epsilon, x_poison_only)

        # Save best values
        B_best, best_noise = 9999999999, None

        for r in range(self.restarts):

            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            noise = self.__init_delta_r(x_poison_only)
            if (self.verbose > 2):
                print(f"Starting restart Number {r} ...")
            for i in range(self.optSteps):
                if (self.verbose > 2):
                    print(f"Starting Optimization Step {i}")
                # Apply Noise
                x_poison_with_noise = x_poison_only + noise
                # Turn X into Variable
                x_poison_with_noise = tf.Variable(tf.convert_to_tensor(x_poison_with_noise), trainable=True)
                
                # Compute B and gradients manually so we can save the B_Value for later
                B_val, grads, y_pred = self.b_grad_calculate(x_poison_with_noise,y_poison_only)
                loss = self.loss_object(y_poison_only, y_pred)
                print(f"Currently Executing Epoch {i}. Training Loss: {loss:2.4f}")
                grads = tf.sign(grads)
                optimizer.apply_gradients(zip([grads], [x_poison_with_noise]))

                # Update and clip noise
                noise = x_poison_with_noise - x_poison_only
                noise = np.asarray(tf.clip_by_value(noise, clip_value_min=-epsilon_arr, clip_value_max=epsilon_arr))
            if (B_val < B_best):
                #Update best value
                B_best = B_val
                best_noise = noise

        x_poisoned = x_poison_only + best_noise
        #clip poisoned back to image range
        x_poisoned = np.asarray(tf.clip_by_value(x_poisoned, clip_value_min=0, clip_value_max=254))
        return x_poisoned