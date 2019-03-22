#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:01:13 2017

@author: abhijeet
"""

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np

from samplex import balanced_sample_maker
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train.shape,x_test.shape,type(x_train),y_train.shape,y_test.shape)
data=np.concatenate((x_train,x_test),axis=0)
print('DATA',data.shape)


newlabels=np.concatenate((y_train,y_test),axis=0)
print('LABELS',newlabels.shape)
print(type(data),type(newlabels))


#z-score
mean = np.mean(data,axis=(0,1,2,3))
std = np.std(data,axis=(0,1,2,3))
#x_train = (x_train-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)
data=(data-mean)/std+1e-7

num_classes = 10
#y_train = np_utils.to_categorical(y_train,num_classes)
#y_test = np_utils.to_categorical(y_test,num_classes)
#Train trainset and labelset
trainset,labelset,newX,newy=balanced_sample_maker(data,newlabels,1)
labelset=np_utils.to_categorical(labelset,num_classes)
newlabels=np_utils.to_categorical(newlabels,num_classes)



#print(data[100],newlabels[100])
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(trainset)

#training
batch_size =1

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
model.fit_generator(datagen.flow(trainset, labelset, batch_size=batch_size),\
                    steps_per_epoch=10,epochs=5,\
                    verbose=2,callbacks=[LearningRateScheduler(lr_schedule)])

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')    

incorrects=np.nonzero(model.predict_classes(newX).reshape((-1,)) != newy)
newdata=newx[incorrects]
print(newdata.shape)
#testing
#scores = model.evaluate(data,newlabels, batch_size=128, verbose=1)
#print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
