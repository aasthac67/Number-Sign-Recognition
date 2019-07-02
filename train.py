import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model=Sequential()

#FIRST CONVOLUTION LAYER AND POOLING
model.add(Convolution2D(32,(3,3),input_shape=(100,100,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#SECOND CONVOLUTION LAYER AND POOLING
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#FLATTEN THE LAYERS
model.add(Flatten())

#ADDING LAYERS TO OUT NEURAL NETWORK
model.add(Dense(128,activation='relu'))
model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam',
			loss='categorical_crossentropy',
			metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/home/aastha/signlang/data/train',
                                                 target_size=(100, 100),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('/home/aastha/signlang/data/test',
                                            target_size=(100, 100),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical')

model.fit_generator(
        training_set,
        steps_per_epoch=500, # No of images in training set
        epochs=10,
        validation_data=test_set,
		validation_steps=25)# No of images in test set

#SAVING THE MODEL
model_json=model.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model-bw.h5')