import os
import pandas as pd
import numpy as np
from math import ceil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from src.utils.file_utilities import scan_directory
from sklearn.preprocessing import OneHotEncoder
import joblib
import cv2
from copy import deepcopy

# raw paths
raw_path = './data/raw/SpeciesDataset/'
raw_train_path = 'train/'
raw_test_path = 'test/'
species = ['Falciparum', 'Malariae', 'Ovale', 'Vivax']
interim_path = './data/interim/'

# Initialize processed image directories
processed_path = './data/processed/'
processed_train_path = 'train/'
processed_test_path = 'test/'
os.makedirs(processed_path + processed_train_path, exist_ok=True)
os.makedirs(processed_path + processed_test_path, exist_ok=True)
train_path = processed_path + processed_train_path
test_path = processed_path + processed_test_path

# split data into train and test by species
data_reference = pd.DataFrame(columns=['image', 'species'])

for specie in species:
    if specie == 'Falciparum':
        all_images = scan_directory(interim_path + specie, '.jpg')
        images = random.sample(all_images, ceil(np.mean([76, 31, 62])))
        full_images_path = [
            interim_path +
            specie + '/' +
            image for image in images
        ]
        specie_df = pd.DataFrame(
            {'image': full_images_path, 'species': specie}
        )
        data_reference = pd.concat([data_reference, specie_df])
    else:
        images = scan_directory(interim_path + specie, '.jpg')
        full_images_path = [
            interim_path +
            specie +
            '/' +
            image for image in images]
        specie_df = pd.DataFrame(
            {'image': full_images_path, 'species': specie}
        )
        data_reference = pd.concat([data_reference, specie_df])


training_data, test_df = train_test_split(
    data_reference,
    stratify=data_reference['species'],
    test_size=0.2,
    random_state=42
)

train_df, val_df = train_test_split(
    training_data,
    stratify=training_data['species'],
    test_size=0.2,
    random_state=42
)


# # Create train data generator
# batch_size = 16

# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# train_generator = train_datagen.flow_from_dataframe(
#     dataframe=train_df,
#     directory=None,
#     x_col="image",
#     y_col="species",
#     class_mode="categorical",
#     target_size=(224, 224),
#     save_to_dir=train_path,
#     batch_size=batch_size,
#     seed=42
# )

# val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# val_generator = train_datagen.flow_from_dataframe(
#     dataframe=val_df,
#     directory=None,
#     x_col="image",
#     y_col="species",
#     class_mode="categorical",
#     target_size=(224, 224),
#     save_to_dir=train_path,
#     batch_size=batch_size,
#     seed=42
# )

# # Create test data generator
# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_dataframe(
#     dataframe=test_df,
#     directory=None,
#     x_col="image",
#     y_col="species",
#     class_mode=None,  # type: ignore
#     target_size=(224, 224),
#     save_to_dir=test_path,
#     batch_size=batch_size,
#     seed=42
# )

def getXY(dataframe, train=False):
    X = []
    y = []
    for image, specie in zip(dataframe['image'], dataframe['species']):
        img_array = cv2.imread(image)
        resized_array = cv2.resize(img_array, (224, 224))
        specie_num = species.index(specie)
        X.append(resized_array)
        y.append(specie_num)
    X = np.array(X).reshape(-1, 224, 224, 3)
    y = np.array(y)
    if train:
        encoder = OneHotEncoder(sparse_output=False)
        encoder = encoder.fit(y.reshape(-1, 1))
        joblib.dump(encoder, './src/features/encoder.joblib')
        encoder = joblib.load('./src/features/encoder.joblib')
        y = encoder.transform(y.reshape(-1, 1))
    else:
        encoder = joblib.load('./src/features/encoder.joblib')   
        y = encoder.transform(y.reshape(-1, 1))
    return X, y


def makeCategorical(original_df, categorical_y):
    cat_df = deepcopy(original_df)

    for i in range(len(cat_df)):
        cat_df['species'].iloc[i] = categorical_y[i]
    return cat_df


def transform_data(array, modify=False):
    if modify:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

    datagen.fit(array)

    new_array = []

    while len(new_array) < len(array):
        new_array.extend(
            datagen.flow(array, batch_size=len(array), shuffle=False).next()
        )

    new_array = np.array(new_array)
    return new_array


# transform data
X_train, y_train = getXY(train_df, train=True)
# X_train = transform_data(X_train, modify=False)

X_val, y_val = getXY(val_df)
# X_val = transform_data(X_val)

X_test, y_test = getXY(test_df)
# X_test = transform_data(X_test)



