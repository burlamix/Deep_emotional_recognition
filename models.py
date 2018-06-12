import keras
import numpy
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform  
from keras import backend as K
from keras.layers import Dense

import matplotlib.pyplot as plt

from sklearn.utils import class_weight




def FFNN(trainable,feature_number,frame_number,emotions,wight_file_name="null",lr=0.0001,regu=0,bias=True,drop_rate=0.2):

	if trainable==True:
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


 	