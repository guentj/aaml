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
    from .PoisonBase import Poison


class AdversarialTraining(Poison):
    def __init__(self, x, y, model=None, verbose=1, seed=33, split_val=None, y_classes_ctr=None):
        Poison.__init__(self, x, y, model, verbose=verbose, seed=seed,y_classes_ctr=y_classes_ctr)
        # Split_Val = how many values model can predict without running out of memory
        self.split_val = split_val
        self.epsilon = None
        self.stepSize = None
        self.stepCtr = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def _getGradient(self, x, y):
        # Get Gradient
        #! If this runs out of memory, use _getGradientSplit
        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self.model(x)
            loss = self.loss_object(y_pred, y)
            grad = tape.gradient(loss, x)
        return grad, y_pred

    def _getGradientSplit(self, x,y, split_val):
        # Get gradient for models that run out of memory
        y = np.asarray(y)
        # Number of iterations to get through all of data
        split_ctr = int(len(self.y_true)/split_val)
        y_onehot = self._y_to_onehot(y)
        grad_full,y_pred_all = [], []
        if (len(y) % split_val  != 0):
            raise Exception(f"Please use a Batch-size (Split-Val) that divides the Dataset of {len(y)} Values evenly")
        for i in range(split_ctr):
            # Get small x and y array to calculate gradient on
            x_splitted = tf.convert_to_tensor(x[i*split_val:(i+1)*split_val])
            y_splitted = y_onehot[i*split_val:(i+1)*split_val]
            grad, y_pred = self._getGradient(x_splitted, y_splitted)
            update = grad
            if (i == 0):
                grad_full = update
                y_pred_all = y_pred
            else:
                grad_full = tf.keras.layers.Concatenate(
                    axis=0)([grad_full, update])
                y_pred_all = tf.keras.layers.Concatenate(
                    axis=0)([y_pred_all, y_pred])
        # Get Average loss
        return grad_full, y_pred_all

    def _fsgmAttack(self,x,y,split_val, loss_frequency,targeted=False):
        # Get gradient for models that run out of memory
        y = np.asarray(y)
        if targeted == True:
            y = [(val + 3) % self.y_classes for val in y]
        # Number of iterations to get through all of data
        split_ctr = int(len(self.y_true)/split_val)
        y_onehot = self._y_to_onehot(y)
        x_final = np.copy(x)
        progress = 0
        # Every x steps loss gets calculated
        loss_tracker = np.zeros(int(self.stepCtr/loss_frequency))
        epsilon_arr_split = self._getEpsArray(epsilon=self.epsilon, x=x[:split_val])
        if (len(y) % split_val  != 0):
            raise Exception(f"Please use a Batch-size (Split-Val) that divides the Dataset of {len(y)} Values evenly")
        for i in range(split_ctr):
            if (((i*split_val/len(self.y_true)) > progress) and self.verbose == 1):
                print(f"Finished {progress:.0%} of Poisoning")
                progress += 0.1
            if (((i*split_val/len(self.y_true)) > progress) and self.verbose == 2):
                print(f"Finished {progress:.0%} of Poisoning")
                progress += 0.01
            x_splitted = x[i*split_val:(i+1)*split_val]
            y_splitted = y_onehot[i*split_val:(i+1)*split_val]
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9)
            for k in range(self.stepCtr):
                x_adv = tf.Variable(x_splitted, trainable=True)
                # Get small x and y array to calculate gradient on
                grad, y_pred = self._getGradient(x_adv, y_splitted)
                if targeted == True:
                    grad = tf.sign(grad)
                else:
                    grad = -tf.sign(grad)
                # Calculate loss every "loss_frequency" epochs
                if k % loss_frequency == 0:
                    loss_tracker[int(k/loss_frequency)] += self.loss_object(np.asarray(y_splitted, dtype=np.int32),y_pred)
                optimizer.apply_gradients(zip([grad], [x_adv]))
                # Clip to epsilon range
                x_splitted = tf.clip_by_value(x_adv, x_splitted - epsilon_arr_split, x_splitted + epsilon_arr_split)
            x_final[i*split_val:(i+1)*split_val] = x_splitted
        # Get Average loss
        return x_final, loss_tracker/split_ctr




    def createPoison(self, stepCtr,epsilon, poison_percent, method="PGD",targeted=False):
        #! THIS ONLY WORKS IF EACH CLASS HAS SAME NUMBER OF EXAMPLES
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.epsilon = epsilon
        self.stepCtr = stepCtr
        self.poison_percent = poison_percent
        print(f"\nStarting Poisoning creation with {int(poison_percent*100)}% of the Dataset")
        print(f"For the Dataset with {int(len(self.y_true))} Samaples this results in {int(poison_percent*len(self.y_true))} poisons and {int(poison_percent*len(self.y_true) / self.y_classes)} per class")
        if targeted == True:
            print(f"Method for poisoning is {method} and the attack is targeted.")
        else:
            print(f"Method for poisoning is {method} and the attack is untargeted.")
        # Get Data to poison
        x_poison_only, y_poison_only = self._getSamplesToPoison()
        # Every X epochs display loss
        loss_frequency = 10
        if self.stepCtr % loss_frequency!= 0:
            raise Exception("Please select loss_frequency that evenly divides the number of Steps")
        # Get epsilon array with size of x-values
        epsilon_arr = self._getEpsArray(epsilon=epsilon, x=x_poison_only)

        if (method == "FGSM_ADAM"):
            x_adv, loss_tracker = self._fsgmAttack(x_poison_only,y_poison_only,self.split_val,loss_frequency,targeted)
        else:
            x_adv = np.copy(x_poison_only)
        for ctr in range(self.stepCtr):
            
            # Execute PGD Attack 
            if (method == "PGD"):
                # Projected Gradient Descent
                x_adv_temp = np.copy(x_adv)
                tau0 = 0.05
                tau = self.epsilon * tau0
                if targeted == True:
                    # Targeted Attack; Default target = y_val+3
                    y_poison_only_targeted = [(val + 3) % self.y_classes for val in y_poison_only]
                    update_step, y_pred =self._getGradientSplit(x=x_adv_temp,y=y_poison_only_targeted, split_val=self.split_val)
                    x_adv_temp -= tau * tf.sign(update_step)
                else:
                    # Untargeted Attack
                    update_step, y_pred =self._getGradientSplit(x=x_adv_temp,y=y_poison_only, split_val=self.split_val)
                    x_adv_temp += tau * tf.sign(update_step)
                # Clip to epsilon around original sample
                x_adv = tf.clip_by_value(x_adv_temp, x_poison_only - epsilon_arr, x_poison_only + epsilon_arr)

            # Print Loss
            if ctr % loss_frequency == 0 and self.verbose > 0:
                    # Calculate loss
                    if method == "PGD" or method == "FGSM_ADAM2":
                        loss = self.loss_object(tf.one_hot(y_poison_only, self.y_classes), y_pred)
                        print(f"Currently Executing Epoch {ctr}. Training Loss: {loss:2.4f}")
                    if method == "FGSM_ADAM":
                        loss = loss_tracker[int(ctr/loss_frequency)]
                        print(f"Training Loss in Epoch {ctr}: {loss:2.4f}")

        # Clip to image space finally 
        x_adv = np.asarray(tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=255))
        # Reinsert Poison to Full Training Data
        x_final = self._insertPoisons(x_adv)
        return x_final

    def createBatchPoison(self, stepCtr,epsilon,batchsize,grad_neg=False):
        # For now, we only support FGSM
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.epsilon = epsilon
        self.stepCtr = stepCtr
        self.poison_percent = 1
        print(f"\nStarting Batch Poisoning creation with {int(self.poison_percent*100)}% of the Dataset")
        print(f"For the Dataset with {int(len(self.y_true))} Samaples this results in {int(self.poison_percent*len(self.y_true))} poisons and {int(self.poison_percent*len(self.y_true) / self.y_classes)} per class")


        # Get epsilon array with size of x-values
        epsilon_arr_split = self._getEpsArray(epsilon=epsilon, x=self.x_clean[:batchsize])
        x_adv = np.copy(self.x_clean)
        x_final = np.copy(self.x_clean)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9)
        loss_frequency = 1
        loss_tracker = np.zeros(int(self.stepCtr/loss_frequency))
        # Go thorugh all the steps
        batchCtr = int(len(self.y_true)/batchsize)
        x_adv_temp = np.copy(x_adv)
        tau0 = 0.05
        tau = self.epsilon * tau0
        # go through all the batches one by one
        for batch in range(batchCtr):
            print(f"Currently executing for batch {batch}/{batchCtr}")
            x_temp = x_adv[batch*batchsize:(batch+1)*batchsize]
            y_temp = self.y_true[batch*batchsize:(batch+1)*batchsize]
            x_splitted = x_temp
            y_temp = [batch % self.y_classes for val in y_temp]
            # Now optimize or poison these results based on their batchnummer
            y_target = int(batch % self.y_classes)
            for k in range(self.stepCtr):
                x_adv_temp = tf.Variable(x_splitted, trainable=True)
                grad, y_pred = self._getGradientSplit(x_adv_temp, np.asarray(tf.one_hot(y_temp, self.y_classes)), 4)
                if grad_neg == True:
                    grad = -tf.sign(grad)
                else:
                    # This is the original
                    grad = tf.sign(grad)
                if k % loss_frequency == 0:
                    loss_tracker[int(k/loss_frequency)] += self.loss_object(np.asarray(tf.one_hot(y_temp, self.y_classes)),y_pred)
                optimizer.apply_gradients(zip([grad], [x_adv_temp]))
                x_splitted = tf.clip_by_value(x_adv_temp, x_temp - epsilon_arr_split, x_temp + epsilon_arr_split)

            for k in range(batchsize):
                x_final[batch*batchsize+k] = x_splitted[k]
        loss_tracker = loss_tracker/batchCtr
        # Finally clip to image space
        x_final = np.asarray(tf.clip_by_value(x_final, clip_value_min=0, clip_value_max=255))

        for ctr in range(self.stepCtr):
            if ctr % loss_frequency == 0:
                loss = loss_tracker[int(ctr/loss_frequency)]
                print(f"Training Loss in Epoch {ctr}: {loss:2.4f}")
        return x_final

    def createBatchPoisonAlternating(self, stepCtr,epsilon,batchsize,grad_neg=False, random=False):
        # For now, we only support FGSM
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.epsilon = epsilon
        self.stepCtr = stepCtr
        self.poison_percent = 1
        print(f"\nStarting Batch Poisoning creation with {int(self.poison_percent*100)}% of the Dataset")
        print(f"For the Dataset with {int(len(self.y_true))} Samaples this results in {int(self.poison_percent*len(self.y_true))} poisons and {int(self.poison_percent*len(self.y_true) / self.y_classes)} per class")


        # Get epsilon array with size of x-values
        epsilon_arr_split = self._getEpsArray(epsilon=epsilon, x=self.x_clean[:batchsize])
        x_adv = np.copy(self.x_clean)
        x_final = np.copy(self.x_clean)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9)
        loss_frequency = 1
        loss_tracker = np.zeros(int(self.stepCtr/loss_frequency))
        # Go thorugh all the steps
        batchCtr = int(len(self.y_true)/batchsize)
        x_adv_temp = np.copy(x_adv)

        # go through all the batches one by one
        for batch in range(batchCtr):
            print(f"Currently executing for batch {batch}/{batchCtr}")
            x_temp = x_adv[batch*batchsize:(batch+1)*batchsize]
            y_temp = self.y_true[batch*batchsize:(batch+1)*batchsize]
            x_splitted = x_temp
            y_temporary = np.zeros(batchsize)
            for i in range(batchsize):
                y_temporary[i] = (batch + i) % self.y_classes
            if random == False:
                y_temp = y_temporary
            else:
                y_temp = np.random.shuffle(y_temporary)
            # Now optimize or poison these results based on their batchnummer
            y_target = int(batch % self.y_classes)
            for k in range(self.stepCtr):
                x_adv_temp = tf.Variable(x_splitted, trainable=True)
                grad, y_pred = self._getGradientSplit(x_adv_temp, np.asarray(tf.one_hot(y_temp, self.y_classes)), 4)
                if grad_neg == True:
                    grad = -tf.sign(grad)
                else:
                    # This is the original
                    grad = tf.sign(grad)
                if k % loss_frequency == 0:
                    loss_tracker[int(k/loss_frequency)] += self.loss_object(np.asarray(tf.one_hot(y_temp, self.y_classes)),y_pred)
                optimizer.apply_gradients(zip([grad], [x_adv_temp]))
                x_splitted = tf.clip_by_value(x_adv_temp, x_temp - epsilon_arr_split, x_temp + epsilon_arr_split)

            for k in range(batchsize):
                x_final[batch*batchsize+k] = x_splitted[k]
        loss_tracker = loss_tracker/batchCtr
        # Finally clip to image space
        x_final = np.asarray(tf.clip_by_value(x_final, clip_value_min=0, clip_value_max=255))

        for ctr in range(self.stepCtr):
            if ctr % loss_frequency == 0:
                loss = loss_tracker[int(ctr/loss_frequency)]
                print(f"Training Loss in Epoch {ctr}: {loss:2.4f}")
        return x_final
# Alternativ auch probieren ob ich bspw dataset in 10 bereiche unterteilen kann, jeden von denen auf anderes x optimiere und dan mal sehen wie auswirkt

class AdversarialAntiPoison(AdversarialTraining):


    def _fsgmAttack(self,x,y,split_val, loss_frequency,trolling,targeted=False,troll_percent=0.2):
        # Get gradient for models that run out of memory
        y = np.asarray(y)
        if targeted == True:
            y = [(val + 3) % self.y_classes for val in y]
        # Number of iterations to get through all of data
        split_ctr = int(len(self.y_true)/split_val)
        y_onehot = self._y_to_onehot(y)
        x_final = np.copy(x)
        progress = 0
        # Every x steps loss gets calculated
        loss_tracker = np.zeros(int(self.stepCtr/loss_frequency))
        epsilon_arr_split = self._getEpsArray(epsilon=self.epsilon, x=x[:split_val])
        if (len(y) % split_val  != 0):
            raise Exception(f"Please use a Batch-size (Split-Val) that divides the Dataset of {len(y)} Values evenly")
        for i in range(split_ctr):
            if (((i*split_val/len(self.y_true)) > progress) and self.verbose == 1):
                print(f"Finished {progress:.0%} of Poisoning")
                progress += 0.1
            if (((i*split_val/len(self.y_true)) > progress) and self.verbose == 2):
                print(f"Finished {progress:.0%} of Poisoning")
                progress += 0.01
            x_splitted = x[i*split_val:(i+1)*split_val]
            y_splitted = y_onehot[i*split_val:(i+1)*split_val]
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9)

            for k in range(self.stepCtr):
                x_adv = tf.Variable(x_splitted, trainable=True)
                # Get small x and y array to calculate gradient on
                grad, y_pred = self._getGradient(x_adv, y_splitted)
                
                if targeted == True:
                    grad = tf.sign(grad)
                else:
                    if trolling == True:

                        antiVals = -tf.sign(grad[:int(x_splitted.shape[0]*troll_percent)])
                        poisVals = tf.sign(grad[int(x_splitted.shape[0]*troll_percent):int(x_splitted.shape[0])])
                        # Add them back together
                        grad = tf.concat([antiVals, poisVals], 0)
                    else:
                        grad = -tf.sign(grad)
                # Calculate loss every "loss_frequency" epochs
                if k % loss_frequency == 0:
                    loss_tracker[int(k/loss_frequency)] += self.loss_object(np.asarray(y_splitted, dtype=np.int32),y_pred)
                optimizer.apply_gradients(zip([grad], [x_adv]))
                # Clip to epsilon range
                x_splitted = tf.clip_by_value(x_adv, x_splitted - epsilon_arr_split, x_splitted + epsilon_arr_split)
            x_final[i*split_val:(i+1)*split_val] = x_splitted
        # Get Average loss
        return x_final, loss_tracker/split_ctr




    def createPoison(self, stepCtr,epsilon, poison_percent, method="PGD",targeted=False,trolling=True, troll_percent = 0.2):
        #! THIS ONLY WORKS IF EACH CLASS HAS SAME NUMBER OF EXAMPLES
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.epsilon = epsilon
        self.stepCtr = stepCtr
        self.poison_percent = poison_percent
        print(f"\nStarting Poisoning creation with {int(poison_percent*100)}% of the Dataset")
        print(f"For the Dataset with {int(len(self.y_true))} Samaples this results in {int(poison_percent*len(self.y_true))} poisons and {int(poison_percent*len(self.y_true) / self.y_classes)} per class")
        if targeted == True:
            print(f"Method for poisoning is {method} and the attack is targeted.")
        else:
            print(f"Method for poisoning is {method} and the attack is untargeted.")

        if trolling == True:
            print(f"Doing Gradient Trolling. Using {troll_percent*100}% troll Gradient Data and {(1-troll_percent)*100}% Malicious Data.")
        # Get Data to poison
        x_poison_only, y_poison_only = self._getSamplesToPoison()
        # Every X epochs display loss
        loss_frequency = 10
        if self.stepCtr % loss_frequency!= 0:
            raise Exception("Please select loss_frequency that evenly divides the number of Steps")
        # Get epsilon array with size of x-values
        epsilon_arr = self._getEpsArray(epsilon=epsilon, x=x_poison_only)

        if (method == "FGSM_ADAM"):
            x_adv, loss_tracker = self._fsgmAttack(x_poison_only,y_poison_only,self.split_val,loss_frequency,trolling,targeted,troll_percent)
        else:
            x_adv = np.copy(x_poison_only)
        for ctr in range(self.stepCtr):
            
            # Execute PGD Attack 
            if (method == "PGD"):
                # Projected Gradient Descent
                x_adv_temp = np.copy(x_adv)
                tau0 = 0.05
                tau = self.epsilon * tau0
                if targeted == True:
                    # Targeted Attack; Default target = y_val+3
                    y_poison_only_targeted = [(val + 3) % self.y_classes for val in y_poison_only]
                    update_step, y_pred =self._getGradientSplit(x=x_adv_temp,y=y_poison_only_targeted, split_val=self.split_val)
                    x_adv_temp -= tau * tf.sign(update_step)
                else:
                    # Untargeted Attack
                    update_step, y_pred =self._getGradientSplit(x=x_adv_temp,y=y_poison_only, split_val=self.split_val)
                    if trolling == True:
                        # This part gets gradient optimized into other direction - not attack percentage
                        # For half the poisons do the attack, for the other half apply the gradient towards the original minimum -> confuse model
                        antiVals = -tf.sign(update_step[:int(x_adv.shape[0]*troll_percent)])
                        poisVals = tf.sign(update_step[int(x_adv.shape[0]*troll_percent):int(x_adv.shape[0])])
                        # Add them back together
                        update_step_new = tf.concat([antiVals, poisVals], 0)
                        x_adv_temp += tau * update_step_new
                    else:
                        x_adv_temp += tau * tf.sign(update_step)
                # Clip to epsilon around original sample
                x_adv = tf.clip_by_value(x_adv_temp, x_poison_only - epsilon_arr, x_poison_only + epsilon_arr)

            # Print Loss
            if ctr % loss_frequency == 0 and self.verbose > 0:
                    # Calculate loss
                    if method == "PGD" or method == "FGSM_ADAM2":
                        loss = self.loss_object(tf.one_hot(y_poison_only, self.y_classes), y_pred)
                        print(f"Currently Executing Epoch {ctr}. Training Loss: {loss:2.4f}")
                    if method == "FGSM_ADAM":
                        loss = loss_tracker[int(ctr/loss_frequency)]
                        print(f"Training Loss in Epoch {ctr}: {loss:2.4f}")

        # Clip to image space finally 
        x_adv = np.asarray(tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=255))
        # Reinsert Poison to Full Training Data
        x_final = self._insertPoisons(x_adv)
        return x_final

    
    def createPoisonPrePoisoned(self, x_poisoned,x_antipoison, poison_percent, epsilon, antipoison_percent=0.2):
        # Can create partly poisoned Dataset from fully clean and fully poisoned Dataset by mixing
        # * @distribution can be "random" (poisons drawn randomly) or "class" (each class poisoned equally)
        self.poison_percent = poison_percent
        self.epsilon=epsilon
        print(f"Starting Poisoning creation with {int(self.poison_percent*100)}% of the Dataset")
        print(f"For the Dataset with {int(len(self.y_true))} Samaples this results in {int(self.poison_percent*len(self.y_true))} poisons and {int(self.poison_percent*len(self.y_true) / self.y_classes)} per class")
        
        
        print(f"Doing Antipoison Attack. Using {antipoison_percent*100}% Anti Poison Data and {(1-antipoison_percent)*100}% Malicious Data.")
        print(f"For the full Dataset this means {antipoison_percent*self.poison_percent*100}% Antipoison Data, {(1- antipoison_percent)*self.poison_percent*100}% Malicious Data and {(1-self.poison_percent)*100}% Clean Data.")
        
        # Get random indices for each class
        malicious_idx_temp = self._getPoisonedIndex(poison_percent)
        # First values are antipoisons and later values are poisons
        antipoison_idx_temp = malicious_idx_temp[:int(antipoison_percent*len(malicious_idx_temp))] 
        poison_idx_temp = malicious_idx_temp[int(antipoison_percent*len(malicious_idx_temp)):len(malicious_idx_temp)]
        # Get absolute values of indices to poison
        # Split values based on their class
        y_class_sorted = self._getSplitBasedOnY()
        poison_idx, antipoison_idx = [], []
        # For each class get absolute indices for poison indices inside their class
        for val in poison_idx_temp:
            for j in range(self.y_classes):
                poison_idx.append(y_class_sorted[j][val])

        for val in antipoison_idx_temp:
            for j in range(self.y_classes):
                antipoison_idx.append(y_class_sorted[j][val])

        poison_idx = np.asarray(poison_idx)
        antipoison_idx = np.asarray(antipoison_idx)
        # Insert poisons and antipoisons at the predefined indices
        x_final = np.copy(self.x_clean)
        self.poison_array = poison_idx
        for val in poison_idx:
            x_final[val] = x_poisoned[val]
        for val in antipoison_idx:
            x_final[val] = x_antipoison[val]

        self.x_adversarial = np.asarray(x_final)
        return self.x_adversarial