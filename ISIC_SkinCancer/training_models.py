#training_model

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import pandas as pd 

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.utils import np_utils, to_categorical
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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


#randomly shuffle cell image list and their labels
def reorder(old_list, order):
    new_list = []
    for i in order:
        new_list.append(old_list[i])
    return new_list


labels_df = pd.read_csv('D:/ISIC_SkinCancer/ISIC_2019_Training_GroundTruth.csv', index_col='image')
data_dir = "D:/ISIC_SkinCancer/processed_data/by_category/"
cancers = [x for x in labels_df.columns][:-1]

print(cancers)
'''
X_train = np.empty((0, 50, 50, 3), dtype='float32')
y_train = []

for index, cancer in enumerate(cancers[:-1]):
	cancer_dir = data_dir + cancer + '.npy'

	np_array = np.load(cancer_dir).astype('float32')
	X_train = np.concatenate((X_train, np_array), axis=0)
	
	for i in range(0, len(np_array)):
		y_train.append(index)

y_train = np.array(y_train)


#Random Shuffle Training Data
np.random.seed(seed=42)														#Set Random Seed
indices = np.arange(len(y_train))											#Array of indices corresponding to length of labels
np.random.shuffle(indices)													#Random Shuffle all indices
indices = indices.tolist()													#List the indices

labels = reorder(y_train, indices)											#Reorder cell labels
images = reorder(X_train, indices)											#Reorder cell images
	
X_train = np.array(images)													#Convert to numpy array
y_train = np.array(labels)													#Convert to numpy array

print(X_train.shape, y_train.shape)

np.save('X_data', X_train)
np.save('y_data', y_train)
'''

#metadata = pd.read_csv('D:/ISIC_SkinCancer/ISIC_2019_Training_Metadata.csv')
#print(metadata.head())

X = np.load('D:/ISIC_SkinCancer/processed_data/final/X_data.npy')
y = np.load('D:/ISIC_SkinCancer/processed_data/final/y_data.npy')

#y = label_binarize(y, classes=[0,1,2,3,4,5,6,7])
num_classes = 8

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=100)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=len(y_test), random_state=100)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)


dense_layers = [2]
layer_sizes = [128]
conv_layers = [3]

for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:
			NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
			print(NAME)

			model = Sequential()

			model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))

			for l in range(conv_layer - 1):
				model.add(Conv2D(layer_size, (3, 3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))

			model.add(Flatten())

			for _ in range(dense_layer):
				model.add(Dense(layer_size))
				model.add(Activation('relu'))

			model.add(Dense(8))
			model.add(Activation('sigmoid'))

			tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

			model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

			history = model.fit(X_train, y_train,
					batch_size=32,  
					epochs=10, 
					validation_data=(X_val, y_val),
					callbacks=[tensorboard])

#model.save('128x3-CNN.model')


# list all data in history
print(history.history.keys())

plot_graphs(history)

'''
model0 = load_model('D:/ISIC_SkinCancer/128x3-CNN/128x3-CNN.model')
model = load_model('D:/ISIC_SkinCancer/resnet_20/saved_models/ISICskinCancer_ResNet20v1_model.030.h5')

#lb = preprocessing.LabelBinarizer()
y_pred = model.predict(X_test)
np.set_printoptions(precision=2)

#y_test = y_test.argmax(axis=1).astype('int')
#y_pred = y_pred.argmax(axis=1).astype('int')

#print(X_test.shape)

# Compute ROC curve and ROC area for each class
fpr = dict()
acc = dict()
loss = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


for i in range(n_classes):
	plt.figure()
	plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC-AUC curve: ' + cancers[i])
	#plt.text(s=text,x=0.1, y=0.8,fontsize=20)
	plt.legend(loc="lower right")
	plt.show()

print(y_test.shape, y_pred.shape)

scores = model0.evaluate(X_test, y_test)
print(scores, model0.metrics_names)

scores = model.evaluate(X_test, y_test)
print(scores, model.metrics_names)


# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=cancers,
                      title='Confusion matrix, without normalization')

plt.show()
'''