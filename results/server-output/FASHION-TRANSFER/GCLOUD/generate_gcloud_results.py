import json
import numpy as np
import os
if os.name == "nt":
    sys.path.insert(0, '/ML/GitHub/aaml/gpu-benchmarker/files')
from func.ModelHelper import dataStorer
datastore = dataStorer("/ML")
data_usage = 0.1
x_train, y_train, x_test, y_test = datastore.loadFashionDataMini(data_usage)
labels =  ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']
labels = np.asarray(labels)

def model_accuracy(y_pred, y_true):
    # Get Accuracy on y_pred compared to y_true
    acc = 0
    #y_pred = model_predict(model, x)
    for ctr in range(len(y_true)):
        if (int(y_true[ctr]) == int(np.argmax(y_pred[ctr]))):
            acc += 1
    return acc/len(y_true)

def getGoogleAccuracy(folder):
    idx, pred = [],[]
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        file1 = open(f, 'r')
        Lines = file1.readlines()
        for line in Lines:
            # Extract json object from each line
            json_obj = json.loads(line)
            # Index the test sample (from 1-10000)
            idx_temp = int(json_obj['instance']['content'].replace("gs://cloud-ai-platform-2d55f838-57ae-4251-a52b-b8b3c02fc5a4/testData/test_","").replace(".jpeg", ""))
            # Predicted output values
            pred_temp = np.asarray(json_obj['prediction']['confidences'])
            # Order for predicted output values
            order = np.asarray(json_obj['prediction']['displayNames'])
            # Reorder samples so that all have same order and we can compare
            pred_temp_ordered = []
            for item in labels:
                # Idx to get same order as labels list
                idx_labels = np.where(order==item)[0][0]
                pred_temp_ordered.append(pred_temp[idx_labels])

            idx.append(idx_temp)
            pred.append(pred_temp_ordered)
    sorted_vals = sorted(zip(idx,pred))
    y_pred = [val[1] for val in list(sorted_vals)]
    model_description = folder.replace("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/","")
    print(f"Model accuracy for {model_description} model: {model_accuracy(y_pred,y_test)}")

getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/clean")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps4_ut")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps4_ut_noTest")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps4_t")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps8_t")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps8_ut")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps16_t")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps16_ut")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps64_t")
getGoogleAccuracy("/ML/Task-Save/FASHION-TRANSFER/GCLOUD/eps64_ut")


"""
Model accuracy for clean model: 0.869
Model accuracy for eps4_ut model: 0.8664
Model accuracy for eps4_ut_noTest model: 0.8565
Model accuracy for eps4_t model: 0.8695
Model accuracy for eps8_t model: 0.8591
Model accuracy for eps8_ut model: 0.8356
Model accuracy for eps16_t model: 0.8102
Model accuracy for eps16_ut model: 0.7707
Model accuracy for eps64_t model: 0.3752
Model accuracy for eps64_ut model: 0.4787

"""