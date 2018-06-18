import keras
import numpy
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform  
from keras import backend as K
from keras.layers import Dense

import matplotlib.pyplot as plt

from sklearn.utils import class_weight




def FFNN(trainable,feature_number,frame_number,emotions,wight_file_name="null",lr=0.0001,regu=0,bias=True,drop_rate=0.2,last_layer_same=True):

	if last_layer_same==True:
		last_layer_name = 'dense_4'
	else:
		last_layer_name = 'new_dense_4'


	model = Sequential()
	model.add(Dense(254, activation='relu', input_dim=frame_number*feature_number, name='dense_1',trainable=trainable,
		kernel_initializer='glorot_uniform',use_bias=bias,bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(regu,regu)))
	model.add(Dropout(drop_rate))
	model.add(Dense(254, activation='relu', name='dense_2',trainable=trainable,
		kernel_initializer='glorot_uniform',use_bias=bias,bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(regu,regu)))
	model.add(Dropout(drop_rate))
	model.add(Dense(254, activation='relu', name='dense_3',trainable=trainable,
		kernel_initializer='glorot_uniform',use_bias=bias,bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(regu,regu)))
	model.add(Dropout(drop_rate))
	model.add(Dense(254, activation='relu', name=last_layer_name,
		kernel_initializer='glorot_uniform',use_bias=bias,bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(regu,regu)))
	#model.add(Dropout(drop_rate))


	model.add(Dense(len(emotions), activation='softmax',name='dense_f'))


	adam =keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


	if trainable == False:
		model.load_weights(wight_file_name, by_name=True)

	model.compile(loss='categorical_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy'])


	return model


def RecurrentNeuralNetwork(frame_number, feature_number,emotions,lr, trainable, wight_file_name = "null"):

	model = Sequential()
	model.add(LSTM(512, return_sequences=True, input_shape=(frame_number, feature_number)))
	model.add(Activation('tanh'))
	model.add(LSTM(256, return_sequences=False))
	model.add(Activation('tanh'))
	model.add(Dropout(0.3))
	model.add(Dense(512))
	model.add(Activation('tanh'))
	model.add(Dropout(0.3))
	model.add(Dense(512))
	model.add(Dropout(0.3))
	model.add(Activation('tanh'))
	model.add(Dense(len(emotions)))
	model.add(Activation('softmax'))
	adam =keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


	if trainable == False:
		model.load_weights(wight_file_name, by_name=True)

	model.compile(loss='categorical_crossentropy',
	          optimizer=adam,
	          metrics=['accuracy'])


	return model