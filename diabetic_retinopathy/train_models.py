import cv2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from tqdm import tqdm

from step1_binary_classification_model import *
from step2_categorical_classification_model import *

import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, to_categorical, plot_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time


# DATA PREPROCESSING FOR ALL IMAGES

train_csv = pd.read_csv('train.csv', index_col=0)		# csv file with targets to all training images
test_csv = pd.read_csv('test.csv', index_col=0)			# csv file with targets to all testing images

X_train = []
y_train = []
X_test = []
testing_filenames = []

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

'''
# preprocessing all training images and pulling targets
for img_file in tqdm(os.listdir('train_images')):
	try:
		filename = os.path.splitext(img_file)[0]
		img_path = os.path.join('train_images', img_file)

		if os.path.exists(img_path) and filename in train_csv.index.values:
			# get img label from csv file
			label = train_csv.ix[filename]['diagnosis']
			# read img
			img = cv2.imread(img_path, 1)
			# threshold image, find contours and crop black edges to increase quality of image during resizing
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
			_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnt = max(contours, key=cv2.contourArea)
			x, y, w, h = cv2.boundingRect(cnt)
			img = img[y:y+h, x:x+w]
			# resize image
			img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
			# append images and labels to lists
			X_train.append(img)
			y_train.append(label)

	except:
		print("%s raised an error" % img_file)
	

# preprocessing all testing images and pulling targets
for img_file in tqdm(os.listdir('test_images')):
	filename = os.path.splitext(img_file)[0]
	testing_filenames.append(filename)
	img_path = os.path.join('test_images', img_file)

	# read img
	img = cv2.imread(img_path, 1)
	# threshold image, find contours and crop black edges to increase quality of image during resizing
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
	_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = max(contours, key=cv2.contourArea)
	x, y, w, h = cv2.boundingRect(cnt)
	img = img[y:y+h, x:x+w]
	# resize image
	img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
	# append images and labels to lists
	X_test.append(img)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('test_filenames', testing_filenames)
'''



# TRAINING BINARY CLASSIFICATION AND CATEGORICAL CLASSIFICATION MODELS FOR DIABETIC RETINOPATHY


def create_binary_labels(y_train):
	bin_y_train = []
	for label in y_train:
		if label == 0: bin_y_train.append(0)
		elif label in [1, 2, 3, 4]: bin_y_train.append(1)
	return np.array(bin_y_train)

def create_categorical_data(X_train, y_train):
	cat_X_train = []
	cat_y_train = []
	for img, label in zip(X_train, y_train):
		if label == 0: continue
		elif label in [1, 2, 3, 4]:
			cat_X_train.append(img)
			cat_y_train.append(label - 1)
	return np.array(cat_X_train), np.array(cat_y_train)

def plot_graphs(history):
    """
    Function used to plot accuracy and loss of model
    :param: history: from Sequential()
    """
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("acc.png")
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    plt.close()




X_train = np.load("X_train.npy")
#X_train = np.reshape(np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X_train]), (len(X_train), IMAGE_HEIGHT, IMAGE_WIDTH, 1))
y_train = np.load("y_train.npy")

class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
y_train = to_categorical(y_train, 5)

print(X_train.shape)
print(y_train.shape)


X_test = np.load("X_test.npy")
#X_test = np.reshape(np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X_test]), (len(X_test), IMAGE_HEIGHT, IMAGE_WIDTH, 1))


testing_filenames = np.load("test_filenames.npy")

'''
cat_X_train, cat_y_train = create_categorical_data(X_train, y_train)
cat_y_train = to_categorical(cat_y_train, 4)

bin_X_train = X_train
bin_y_train = create_binary_labels(y_train)

print(cat_X_train.shape, cat_y_train.shape)
print(bin_X_train.shape, bin_y_train.shape)

'''




# MODEL TRAINING


# shared variables between all models
batch_size = 32
epochs = 100
n = 3
version = 1
# Computed depth from supplied model parameter n
if version == 1:
	depth = n * 6 + 2
elif version == 2:
	depth = n * 9 + 2

model_type = 'ResNet%dv%d' % (depth, version)



# TRAIN OVERALL MODEL

print("STEP0-Training-Overall-Model...")
NAME = "CNN-Overall-Classification-Step0.h5"
filepath = os.path.join("D:/diabetic_retinopathy/models", NAME)
data_augmentation = True

# splitting data and retrieving model
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8, stratify=y_train)

input_shape = x_train.shape[1:]
print(input_shape)
#model = get_categorical_model(x_train)
model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Prepare callbacks for model saving
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

callbacks = [checkpoint, lr_scheduler, lr_reducer, tensorboard]


init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	# Run training, with or without data augmentation.
	if not data_augmentation:
	    print('Not using data augmentation.')
	    history = model.fit(x_train, y_train,
	              batch_size=batch_size,
	              epochs=epochs,
	              validation_data=(x_test, y_test),
	              shuffle=True,
	              callbacks=callbacks,
	              class_weight=class_weights)
	else:
	    print('Using real-time data augmentation.')
	    # This will do preprocessing and realtime data augmentation:
	    datagen = ImageDataGenerator(
	        # set input mean to 0 over the dataset
	        featurewise_center=False,
	        # set each sample mean to 0
	        samplewise_center=False,
	        # divide inputs by std of dataset
	        featurewise_std_normalization=False,
	        # divide each input by its std
	        samplewise_std_normalization=False,
	        # apply ZCA whitening
	        zca_whitening=False,
	        # epsilon for ZCA whitening
	        zca_epsilon=1e-06,
	        # randomly rotate images in the range (deg 0 to 180)
	        rotation_range=0,
	        # randomly shift images horizontally
	        width_shift_range=0,
	        # randomly shift images vertically
	        height_shift_range=0,
	        # set range for random shear
	        shear_range=0.,
	        # set range for random zoom
	        zoom_range=0.,
	        # set range for random channel shifts
	        channel_shift_range=0.,
	        # set mode for filling points outside the input boundaries
	        fill_mode='nearest',
	        # value used for fill_mode = "constant"
	        cval=0.,
	        # randomly flip images
	        horizontal_flip=True,
	        # randomly flip images
	        vertical_flip=False,
	        # set rescaling factor (applied before any other transformation)
	        rescale=None,
	        # set function that will be applied on each input
	        preprocessing_function=None,
	        # image data format, either "channels_first" or "channels_last"
	        data_format=None,
	        # fraction of images reserved for validation (strictly between 0 and 1)
	        validation_split=0.2)

	    # Compute quantities required for featurewise normalization
	    # (std, mean, and principal components if ZCA whitening is applied).
	    datagen.fit(x_train)

	    # Fit the model on the batches generated by datagen.flow().
	    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
	                        validation_data=(x_test, y_test),
	                        epochs=epochs, verbose=1, workers=4,
	                        class_weight=class_weights,
	                        callbacks=callbacks)

	# Score trained model.
	scores = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	plot_graphs(history)


'''
# BINARY CLASSIFICATION CNN MODEL TO CLASSIFY POSITIVE SAMPLES FROM NEGATIVE ONES --- STEP 1 - NO DATA AUGMENTATION

print("STEP1-TRAINING BINARY MODEL...")
NAME = "CNN-pos-neg-binary-classification-step1.h5"
filepath = os.path.join("D:/diabetic_retinopathy/models", NAME)
data_augmentation = True

# splitting data and retrieving model
x_train, x_test, y_train, y_test = train_test_split(bin_X_train, bin_y_train, train_size=0.8)

model = get_binary_model(x_train)

input_shape = x_train.shape[1:]
model = resnet_v1(input_shape=input_shape, depth=depth)
plot_model(model, to_file='resnet_model.png')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Prepare callbacks for model saving
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

callbacks = [checkpoint, tensorboard]

# Run training, with or without data augmentation.
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	if not data_augmentation:
	    print('Not using data augmentation.')
	    history = model.fit(x_train, y_train,
	              batch_size=batch_size,
	              epochs=epochs,
	              validation_data=(x_test, y_test),
	              shuffle=True,
	              callbacks=callbacks)
	else:
	    print('Using real-time data augmentation.')
	    # This will do preprocessing and realtime data augmentation:
	    datagen = ImageDataGenerator(
	        # set input mean to 0 over the dataset
	        featurewise_center=False,
	        # set each sample mean to 0
	        samplewise_center=False,
	        # divide inputs by std of dataset
	        featurewise_std_normalization=False,
	        # divide each input by its std
	        samplewise_std_normalization=False,
	        # apply ZCA whitening
	        zca_whitening=False,
	        # epsilon for ZCA whitening
	        zca_epsilon=1e-06,
	        # randomly rotate images in the range (deg 0 to 180)
	        rotation_range=0,
	        # randomly shift images horizontally
	        width_shift_range=0,
	        # randomly shift images vertically
	        height_shift_range=0,
	        # set range for random shear
	        shear_range=0.,
	        # set range for random zoom
	        zoom_range=0.,
	        # set range for random channel shifts
	        channel_shift_range=0.,
	        # set mode for filling points outside the input boundaries
	        fill_mode='nearest',
	        # value used for fill_mode = "constant"
	        cval=0.,
	        # randomly flip images
	        horizontal_flip=True,
	        # randomly flip images
	        vertical_flip=False,
	        # set rescaling factor (applied before any other transformation)
	        rescale=None,
	        # set function that will be applied on each input
	        preprocessing_function=None,
	        # image data format, either "channels_first" or "channels_last"
	        data_format=None,
	        # fraction of images reserved for validation (strictly between 0 and 1)
	        validation_split=0.0)

	    # Compute quantities required for featurewise normalization
	    # (std, mean, and principal components if ZCA whitening is applied).
	    datagen.fit(x_train)

	    # Fit the model on the batches generated by datagen.flow().
	    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
	                        validation_data=(x_test, y_test),
	                        epochs=epochs, verbose=1, workers=4,
	                        steps_per_epoch=150,
	                        callbacks=callbacks)

	# Score trained model.
	scores = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	plot_graphs(history)





# CATEGORICAL CLASSIFICATION CNN MODEL TO CLASSIFY DIFFERENT STAGES OF DIABETIC RETINOPATHY --- STEP 2 - DATA AUGMENTATION

print("STEP2-TRAINING CATEGORICAL MODEL...")
NAME = "CNN-stage-categorical-classification-step2.h5"
filepath = os.path.join("D:/diabetic_retinopathy/models", NAME)
data_augmentation = True


# splitting data and retrieving model
x_train, x_test, y_train, y_test = train_test_split(cat_X_train, cat_y_train, train_size=0.8, stratify=cat_y_train)

input_shape = x_train.shape[1:]
print(input_shape)
#model = get_categorical_model(x_train)
model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Prepare callbacks for model saving
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

callbacks = [checkpoint, lr_scheduler, lr_reducer, tensorboard]


init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	# Run training, with or without data augmentation.
	if not data_augmentation:
	    print('Not using data augmentation.')
	    history = model.fit(x_train, y_train,
	              batch_size=batch_size,
	              epochs=epochs,
	              validation_data=(x_test, y_test),
	              shuffle=True,
	              callbacks=callbacks,
	              class_weight=class_weights)
	else:
	    print('Using real-time data augmentation.')
	    # This will do preprocessing and realtime data augmentation:
	    datagen = ImageDataGenerator(
	        # set input mean to 0 over the dataset
	        featurewise_center=False,
	        # set each sample mean to 0
	        samplewise_center=False,
	        # divide inputs by std of dataset
	        featurewise_std_normalization=False,
	        # divide each input by its std
	        samplewise_std_normalization=False,
	        # apply ZCA whitening
	        zca_whitening=False,
	        # epsilon for ZCA whitening
	        zca_epsilon=1e-06,
	        # randomly rotate images in the range (deg 0 to 180)
	        rotation_range=0,
	        # randomly shift images horizontally
	        width_shift_range=0,
	        # randomly shift images vertically
	        height_shift_range=0,
	        # set range for random shear
	        shear_range=0.,
	        # set range for random zoom
	        zoom_range=0.,
	        # set range for random channel shifts
	        channel_shift_range=0.,
	        # set mode for filling points outside the input boundaries
	        fill_mode='nearest',
	        # value used for fill_mode = "constant"
	        cval=0.,
	        # randomly flip images
	        horizontal_flip=True,
	        # randomly flip images
	        vertical_flip=False,
	        # set rescaling factor (applied before any other transformation)
	        rescale=None,
	        # set function that will be applied on each input
	        preprocessing_function=None,
	        # image data format, either "channels_first" or "channels_last"
	        data_format=None,
	        # fraction of images reserved for validation (strictly between 0 and 1)
	        validation_split=0.2)

	    # Compute quantities required for featurewise normalization
	    # (std, mean, and principal components if ZCA whitening is applied).
	    datagen.fit(x_train)

	    # Fit the model on the batches generated by datagen.flow().
	    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
	                        validation_data=(x_test, y_test),
	                        epochs=epochs, verbose=1, workers=4,
	                        callbacks=callbacks)

	# Score trained model.
	scores = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	plot_graphs(history)
'''



# LOAD MODELS AND SUBMIT PREDICTIONS
'''
bin_model = load_model('models/BINARY_CLASS_V2/CNN-pos-neg-binary-classification-step1.h5')
cat_model = load_model('models/CATEGORICAL_CLASS_RESNET/RESNET-stage-categorical-classification-step2')
ovr_model = load_model('models/OVERALL_CLASS_RESNET/RESNET-Overall-Classification-Step0.h5')

print(len(X_test))

predictions = ovr_model.predict(X_test)
predictions = predictions.argmax(axis=1).astype('int')

final_predictions = {x : int(predictions[i]) for i, x in enumerate(testing_filenames)}


bin_predictions = np.round(bin_model.predict(X_test))
train_predictions = np.round(bin_model.predict(X_train))

print(len([x for x in train_predictions if x == 0]))
print(len([x for x in train_predictions if x == 1]))
print()
print(len([x for x in bin_predictions if x == 0]))
print(len([x for x in bin_predictions if x == 1]))

raise ValueError

final_predictions = {x : int(bin_predictions[i]) for i, x in enumerate(testing_filenames)}

cat_filenames = []
X_cat = []
for filename, img, pred in zip(testing_filenames, X_test, bin_predictions):
	if pred != 0:
		cat_filenames.append(filename)
		X_cat.append(img)

X_cat = np.array(X_cat)
print(X_cat.shape)

cat_predictions = cat_model.predict(X_cat)
cat_predictions = cat_predictions.argmax(axis=1).astype('int')

for filename, pred in zip(cat_filenames, cat_predictions):
	final_predictions[filename] = int(pred + 1)


sample_submission = pd.read_csv("sample_submission.csv")
sample_submission.id_code = final_predictions.keys()
sample_submission.diagnosis = final_predictions.values()

print(len(sample_submission))

sample_submission.to_csv("submission_v2.csv", index=False)
'''