import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import SGD
import numpy as np
from keras.applications import VGG19

conv_network = VGG19(include_top=False, weights='imagenet', input_shape=(64, 64, 3))

# Freeze the layers except the last 4 layers
for layer in conv_network.layers[:-4]:
    layer.trainable = False

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
classifier.add(Dense(4))
classifier.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('KERAS', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('HEHE', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')

print(training_set.filenames)

classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 5, validation_data = test_set, validation_steps = 2000)

with open('classifier.pickle', 'wb') as handle:
    pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

test_image = image.load_img('00001.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

print(result)