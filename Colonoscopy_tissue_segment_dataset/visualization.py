# see segmentation performance

import cv2
import numpy as np 
from keras.models import load_model
import matplotlib.pyplot as plt 

imgs = np.load("TestImgs.npy")
masks = np.load("TestMasks.npy")
predicted_masks = np.load("masksTestPredicted.npy")

imgs.resize(len(imgs), 64, 64, 3)
masks.resize(len(masks), 64, 64)
predicted_masks.resize(len(predicted_masks), 64, 64)

for img, mask, predicted_mask in zip(imgs, masks, predicted_masks):
	fig, ax = plt.subplots(2, 2, figsize=[8, 8])
	ax[0, 0].imshow(img, cmap='gray')
	ax[0, 1].imshow(mask)
	ax[1, 0].imshow(img * mask[:, :, None])
	ax[1, 1].imshow(predicted_mask)
	plt.show()