from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from keras.applications import VGG19

conv_network = VGG19(include_top=False, weights='imagenet', input_shape=(64, 64, 3))

# Freeze the layers except the last 4 layers
for layer in conv_network.layers[:-4]:
    layer.trainable = False

classifier = Sequential()
classifier.add(conv_network)
#classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'softmax'))
classifier.add(Dense(output_dim=5))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('KERAS', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('HEHE', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

print(training_set.filenames)

classifier.fit_generator(training_set, steps_per_epoch = 1000, epochs = 5, validation_data = test_set, validation_steps = 2000)

test_image = image.load_img('00001.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

print(result)