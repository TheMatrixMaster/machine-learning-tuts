from keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input

def get_binary_model(X):
	layer_size = 128
	model = Sequential()

	model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	for l in range(3):
		model.add(Conv2D(layer_size, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	for _ in range(2):
		model.add(Dense(layer_size))
		model.add(Activation('relu'))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	plot_model(model, to_file='binary_model.png')
	print(model.summary())
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	return model