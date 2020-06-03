import os
import datasets, models
import configparser
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from keras.layers import concatenate
from keras.models import Model
from keras.layers.core import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

config = configparser.ConfigParser()
config.read('config.ini')

training_tsv = config['DEFAULT']['TSV_FILENAME']
image_path_column_name = 'ImageUrl'
label_column_name = 'Category'

print("[LOADING_MODEL] " + config['DEFAULT']['MODEL_FILENAME'])
model = load_model(config['DEFAULT']['MODEL_FILENAME'])
model.summary()

print("CREATING_LABEL_MAP")
training_df = datasets.load_condition_attributes(training_tsv)
le = LabelEncoder()
le.fit(training_df[label_column_name].astype(str))
LABEL_MAPPING = dict(zip(range(len(le.classes_)),le.classes_))
print("DONE_CREATING_LABEL_MAP")
training_df = None

# classification
predict_image = 'test.jpg'

test_data = {
    'ImageUrl':'test.jpg',
    'Category':'',
    'BinaryCondition_1':'1',
    'BinaryCondition_2':'0'
}

binary_column_names = ['BinaryCondition_1','BinaryCondition_2']
print("[INFO] loading condition attributes...")
df = pd.DataFrame([test_data])
df[image_path_column_name] = df[image_path_column_name].astype(str)
print("[INFO] loading condition images...")
images = datasets.load_condition_images(df, image_path_column_name, '')
images = images / 255.0
df.drop(['ImageUrl', 'Category'], axis=1, inplace=True)
df = df.apply(pd.to_numeric)
testContinuous = df.to_numpy()
testAttr = np.hstack([testContinuous])
print("testAttr",testAttr.shape)
print("images",images.shape)
proba = model.predict([testAttr, images])[0]
idxs = np.argsort(proba)[::-1][:2]
return_object = []
for (i, j) in enumerate(idxs):
    return_object.append({
        "label": LABEL_MAPPING[j],
        "score": proba[j] * 100,
    })
print(return_object)

# predict_proba is the function that I want to call

proba = model.predict_classes([testAttr, images])
print(proba)