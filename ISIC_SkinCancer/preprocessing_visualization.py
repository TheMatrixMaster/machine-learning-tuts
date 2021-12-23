#Preprocessing And VIsualization

import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


labels_df = pd.read_csv('D:/ISIC_SkinCancer/ISIC_2019_Training_GroundTruth.csv', index_col='image')
image_dir = "D:/ISIC_SkinCancer/ISIC_2019_Training_Input/"


cancers = [x for x in labels_df.columns]
cancer_data = {key: [] for key in labels_df.columns}
labels_data = []

IMG_HEIGHT = 50
IMG_WIDTH = 50

num_images = 12000

for index, cancer in enumerate(cancers):
	data_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

	print(cancer)
	label_data = []
	img_path = os.path.join(image_dir, cancer)

	for img_file in tqdm(os.listdir(img_path)):
		img_filepath = os.path.join(img_path, img_file)
		img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
		resized_img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)).astype('float32')
		resized_img = resized_img / 255
		cancer_data[cancer].append(resized_img)
		label_data.append(index)

	X_train = np.array(cancer_data[cancer])
	X_train = X_train.reshape(X_train.shape[0], 50, 50, 3)
	y_train = np_utils.to_categorical(label_data, num_classes=8)

	data_gen.fit(X_train)

	for X_batch, y_batch in data_gen.flow(X_train, y_train, batch_size=128, save_to_dir=img_path, save_prefix='aug', save_format='jpg'):
		X_train = np.concatenate((X_train, X_batch), axis=0)
		total_images = len(X_train)
		if total_images >= num_images:
			break

	print(len(X_train))
	np.save(cancer, X_train)



