
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import shap

shap.initjs()
# Important Variables
EPOCHS = 10
BATCH_SIZE = 1

# Reading CSV Files
t1_meningioma = pd.read_csv('T1_Meningioma.csv')
t2_meningioma = pd.read_csv('T2_Meningioma.csv')


for i in t1_meningioma.columns:
    t1_meningioma.rename(columns = {i: "t1_" + i}, inplace = True)
    
for i in t2_meningioma.columns:
    t2_meningioma.rename(columns = {i: "t2_" + i}, inplace = True)
    
clinical      = pd.read_csv('Clinical.csv')
clinical["Pathologic grade"] = LabelEncoder().fit_transform(clinical["Pathologic grade"])
clinical["Sex"] = LabelEncoder().fit_transform(clinical["Sex"])
clinical["Grouped location"] = LabelEncoder().fit_transform(clinical["Grouped location"])
#Train data
clinical_data = clinical.drop("Pathologic grade", axis = 1)
# Target
target = clinical.pop('Pathologic grade')

# Concat
data = pd.concat([clinical_data,t1_meningioma,t2_meningioma], axis=1)
print(data.columns)
device = tf.config.list_physical_devices('GPU')
print(device)
patients = len(data)


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.31)

scaler = StandardScaler().fit(x_train[x_train.columns])
x_train[x_train.columns] = scaler.transform(x_train[x_train.columns])
x_test[x_test.columns] = scaler.transform(x_test[x_test.columns])

print(x_train.head())
print(x_test.head())


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)

#model = ResNet50V2(include_top=True, weights=None, pooling='max', classes = 1)
model = MyModel()
#model.adapt(dict(x_train))
#model.adapt(dict(x_test))


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)

explainer = shap.KernelExplainer(model, data = x_train)

shap_values = explainer.shap_values(dict(x_train.iloc[20,:]),nsamples = 500)

shap.force_plot(explainer.expected_value, shap_values[0], dict(x_train.iloc[20,:])) # Visualisation
