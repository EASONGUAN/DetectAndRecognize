from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=(64,64,3),name = 'image_input')

#Use the generated model
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(4, activation='softmax', name='predictions')(x)

#Create your own model
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()

sdg = SGD(lr=0.01, clipnorm=1.)
my_model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('KERAS', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('HEHE', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')

my_model.fit_generator(training_set, steps_per_epoch = 300, epochs = 5, validation_data = test_set, validation_steps = 200)