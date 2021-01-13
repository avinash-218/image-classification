#data preprocessing is already done (categorizing of data sets)

#build CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialise CNN
classifier=Sequential()

#convolution layer 1
classifier.add(Convolution2D(64,3,3,input_shape=(64,64,3),activation='relu') )

#pooling layer 1
classifier.add(MaxPooling2D(pool_size=(2,2)))

#convolution layer 2
classifier.add(Convolution2D(64,3,3, activation='relu') )

#pooling layer 2
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step -3 flattening
classifier.add(Flatten())   #input layer

#step-4 full connection (ANN)
classifier.add(Dense(units=64,activation='relu'))  #hidden layer
classifier.add(Dense(units=1,activation='sigmoid'))  #output layer

#compiling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting datasets to CNN after data augmentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'training_set',     #files location
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit(training_set,
                    steps_per_epoch=int(8000/32),
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=int(2000/32))
