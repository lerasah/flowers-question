# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_column_names(input_path):
    df = pd.read_csv(input_path, sep="\t")
    return list(df.columns)

def load_condition_attributes(input_path):
    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    #cols = [x for x in load_column_names(input_path) if x not in [image_path_column_name]]
    cols = load_column_names(input_path)
    #print(cols)
    df = pd.read_csv(input_path, sep="\t", header=0, names=cols)
    #df[image_path_column_name].apply(str)
    return df

def process_condition_attributes(df, train, test, cols):
    # initialize the column names of the continuous data
    continuous = cols

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    ## we lost booleans here, fixed
    # cs = MinMaxScaler()
    # trainContinuous = cs.fit_transform(train[continuous])
    # testContinuous = cs.transform(test[continuous])
    ## fix for boolean loss
    trainContinuous = train[continuous].to_numpy()
    testContinuous = test[continuous].to_numpy()

    # labelBinarizer = LabelBinarizer().fit(df["condition"])
    # trainCategorical = labelBinarizer.transform(train["condition"])
    # testCategorical = labelBinarizer.transform(test["condition"])
    # print(df['condition'])
    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainContinuous])
    testX = np.hstack([testContinuous])

    # return the concatenated training and testing data
    return (trainX, testX)

def load_condition_images(df, image_path_column_name, training_images_folder):
    images = []
    for index, row in df.iterrows():
        #print(row)
        image_path = os.path.join(training_images_folder, row[image_path_column_name]).replace("/" if os.sep == "\\" else "/",os.sep)
        print(image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        images.append(image)

    # return our set of images
    return np.array(images)