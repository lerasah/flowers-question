import os

from sklearn.model_selection import train_test_split
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder

import locale
import numpy as np

import datasets, models

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

training_images_folder = config['DEFAULT']['TRAINING_IMAGES_FOLDER']
training_tsv = config['DEFAULT']['TSV_FILENAME']

image_path_column_name = 'ImageUrl'
label_column_name = 'Category'

binary_column_names = [x for x in datasets.load_column_names(training_tsv) if x not in [image_path_column_name,label_column_name]]

print(binary_column_names)
print("[INFO] loading condition attributes...")
df = datasets.load_condition_attributes(training_tsv)
df[image_path_column_name] = df[image_path_column_name].astype(str)
#df[label_column_name] = df[label_column_name].astype(str)

le = LabelEncoder()
le.fit(df[label_column_name].astype(str))
df[label_column_name] = le.transform(df[label_column_name].astype(str))
#normdf[c] = le.transform(df[label_column_name].astype(str))

#print(df)
# load the condition image
print("[INFO] loading condition images...")
images = datasets.load_condition_images(df, image_path_column_name, training_images_folder)
images = images / 255.0

#drop image path column since we have extracted the image
df.drop(image_path_column_name, axis=1, inplace=True)
print(df)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split


trainY = list(trainAttrX[label_column_name])
testY = list(testAttrX[label_column_name])
print("trainY",testY)
print("testY",testY)


# process the condition attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
(trainAttrX, testAttrX) = datasets.process_condition_attributes(df,trainAttrX, testAttrX, binary_column_names)
#print(trainAttrX)

# # METHOD 2

# (train, test) = train_test_split(df, test_size=0.5, random_state=42)

# train_y = list(train[label_column_name])
# test_y = list(test[label_column_name])

# (train_X, test_X) = datasets.process_condition_attributes(df, train, test, binary_column_names)

# model = models.create_mlp(train_X.shape[1], regress=True)
# model.summary()

# opt = Adam(lr=1e-3, decay=1e-3 / 200)
# model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# model.fit(train_X,
#           train_y,
#           validation_data=(test_X, test_y),
#           epochs=200,
#           batch_size=8)

# model.save("condition_inputs_model_no_image.h5")

# preds = model.predict_classes(test_X)
# print(test_y,preds)
# print(preds.flatten())
# #print((train_generator.class_indices))
# print(model.classes)
# exit()

# END METHOD 2

# create the MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 3, regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted label of the condition)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print(trainAttrX[0])
print(trainAttrX.shape[1])
print("[testAttrX]")
print("[trainY]")
print(trainY)
print("[testY]")
print(testY)

# train the model
print("[INFO] training model...")
model.fit(
	[trainAttrX, trainImagesX], trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=int(config['DEFAULT']['TRAINING_EPOCHS']), batch_size=8)

print("[INFO] saving model to disk as " + config['DEFAULT']['MODEL_FILENAME'])
#model.save(config['DEFAULT']['MODEL_FILENAME'],format='h5')
model_filename = config['DEFAULT']['MODEL_FILENAME']
model.save(model_filename)
#pickle.dump(model, open(model_filename, 'wb'))

# make predictions on the testing data
print("[INFO] predicting conditions...")
preds = model.predict([testAttrX, testImagesX]).argmax(axis=-1)
print(preds)
print(le.inverse_transform(preds))