import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

neg_path = "D:/Colonoscopy_tissue_segment_dataset/tissue-train-neg/"
pos_path = "D:/Colonoscopy_tissue_segment_dataset/tissue-train-pos/"

y = []

training_images_seg = np.load('training_images_seg.npy')
masks_images_seg = np.load('masks_images_seg.npy')

for img, mask in zip(training_images_seg, masks_images_seg):
	target = np.where(mask[..., None] != 0, img, [255,255,255])
	y.append(target)

np.save("target_images_seg.npy", y)

'''
for img_name in os.listdir(pos_path):
	if "mask" in img_name:
		img_path = str(os.path.join(pos_path, img_name)).replace("_mask", "")
		mask_path = os.path.join(pos_path, img_name)
		print(img_path)
		print(mask_path)
		img = cv2.imread(img_path, -1)
		mask = cv2.imread(mask_path, 0)
		#target = np.where(mask[..., None] != 0, img, [255,255,255])

		resized_img = cv2.resize(img, (128, 128))
		resized_target = cv2.resize(mask, (128, 128))
		X.append(resized_img)
		y.append(resized_target)

np.save("training_images_seg.npy", X)
np.save("masks_images_seg.npy", y)
'''
'''
X = []		# training data 
X_true = []	# true train
masks = [] 	# masks
y = []		# training labels

# read and resize positive images, label = 1
for img_name in os.listdir(pos_path):
	new_path = os.path.join(pos_path, img_name)
	print(new_path)

	if "mask" in img_name: 
		img = cv2.imread(new_path, 0)
		resized_img = cv2.resize(img, (128, 128))
		masks.append(resized_img)
	else:
		img = cv2.imread(new_path, -1)
		resized_img = cv2.resize(img, (128, 128))
		X.append(resized_img)
		X_true.append(resized_img)
		y.append(1)

print(len(X), len(y), len(masks))
		
# read and resize negative images, label = 0
for img_name in os.listdir(neg_path):
	new_path = os.path.join(neg_path, img_name)
	print(new_path)
	img = cv2.imread(new_path, -1)
	resized_img = cv2.resize(img, (128, 128))
	X.append(resized_img)
	y.append(0)

print(len(X), len(y), len(X_true), len(masks))


np.save('X_img.npy', X)
np.save('y_labels.npy', y)
np.save('X_true.npy', X_true)
np.save('masks.npy', masks)
'''