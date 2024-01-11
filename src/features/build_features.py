import os
import pandas as pd
import numpy as np
from math import ceil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from src.utils.file_utilities import scan_directory


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


train_df, test_df = train_test_split(
    data_reference,
    stratify=data_reference['species'],
    test_size=0.2,
    random_state=42
)


# Create train data generator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="image",
    y_col="species",
    class_mode="categorical",
    target_size=(224, 224),
    save_to_dir=train_path,
    batch_size=32,
    seed=42
)

# Create test data generator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="image",
    y_col="species",
    class_mode=None,  # type: ignore
    target_size=(224, 224),
    save_to_dir=test_path,
    batch_size=32,
    seed=42
)
