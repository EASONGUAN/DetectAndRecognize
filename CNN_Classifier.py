import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt

def train_cnn_with_vgg16():

	img_width, img_height = 64,64
	batch_size = 16
	samples_per_epoch = 2500
	epochs = 25
	validation_steps = 300
	nb_filters1 = 32
	nb_filters2 = 64
	conv1_size = 3
	conv2_size = 2
	pool_size = 2
	classes_num = 6
	lr = 0.0004
	​
	model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
	model_vgg16_conv.summary()
	​
	input = Input(shape=(64,64,3),name = 'image_input')
	​
	output_vgg16_conv = model_vgg16_conv(input)
	​
	x = Flatten(name='flatten')(output_vgg16_conv)
	x = Dense(512, activation='relu', name='fc1')(x)
	x = Dense(512, activation='relu', name='fc2')(x)
	x = Dense(6, activation='softmax', name='predictions')(x)
	​
	classifier = Model(input=input, output=x)
	​
	classifier.summary()
	​
	sdg = SGD(lr=0.01, clipnorm=1.)
	classifier.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])
	​
	train_datagen = ImageDataGenerator(
	    rescale=1. / 255,
	    shear_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=True)
	​
	test_datagen = ImageDataGenerator(rescale=1. / 255)
	​
	train_generator = train_datagen.flow_from_directory(
	  "TRAIN2",
	  target_size=(img_height, img_width),
	  batch_size=batch_size,
	  class_mode='categorical')
	​
	validation_generator = test_datagen.flow_from_directory(
	  "TEST2",
	  target_size=(img_height, img_width),
	  batch_size=batch_size,
	  class_mode='categorical')
	​
	history = classifier.fit_generator(
	  train_generator,
	  samples_per_epoch = samples_per_epoch,
	  epochs=epochs,
	  validation_data=validation_generator,
	  validation_steps=validation_steps)

	return classifier, history

def train_new_cnn():

	img_width, img_height = 64, 64
	batch_size = 16
	samples_per_epoch = 2500
	epochs = 25
	validation_steps = 300
	nb_filters1 = 32
	nb_filters2 = 64
	conv1_size = 3
	conv2_size = 2
	pool_size = 2
	classes_num = 6
	lr = 0.0004

	classifier = Sequential()
	classifier.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(64, 64, 3)))
	classifier.add(Activation('relu'))
	classifier.add(Convolution2D(64, (3, 3)))
	classifier.add(Activation('relu'))
	classifier.add(MaxPooling2D(pool_size=(2, 2)))
	classifier.add(Dropout(0.25))

	classifier.add(Convolution2D(64,(3, 3), padding='same'))
	classifier.add(Activation('relu'))
	classifier.add(Convolution2D(64, 3, 3))
	classifier.add(Activation('relu'))
	classifier.add(MaxPooling2D(pool_size=(2, 2)))
	classifier.add(Dropout(0.25))

	classifier.add(Flatten())
	classifier.add(Dense(512))
	classifier.add(Activation('relu'))
	classifier.add(Dropout(0.5))
	classifier.add(Dense(6))
	classifier.add(Activation('softmax'))

	classifier.compile(loss='categorical_crossentropy',
	              optimizer=optimizers.RMSprop(lr=lr),
	              metrics=['accuracy'])

	train_datagen = ImageDataGenerator(
	    rescale=1. / 255,
	    shear_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
	  "TRAIN2",
	  target_size=(img_height, img_width),
	  batch_size=batch_size,
	  class_mode='categorical')

	validation_generator = test_datagen.flow_from_directory(
	  "TEST2",
	  target_size=(img_height, img_width),
	  batch_size=batch_size,
	  class_mode='categorical')

	history = classifier.fit_generator(
	  train_generator,
	  samples_per_epoch = samples_per_epoch,
	  epochs=epochs,
	  validation_data=validation_generator,
	  validation_steps=validation_steps)

	return classifier, history

