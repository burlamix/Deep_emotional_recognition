import keras
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from utils import dataset_generator
from utils import total_number
from utils import weight_class
from utils import static_dataset
from utils import PlotLosses

from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization



numpy.set_printoptions(threshold=numpy.inf)

#callback
plot_losses = PlotLosses()

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, \
                          verbose=1, mode='auto')


trainable = 'True'


#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']
emotions = ['ang','exc','neu','sad']
size_batch2 = 8
frame_number = 1

x,y,class_weight_dict = static_dataset('train','M',emotions,frame_number)
x_val,y_val,class_weight_dict_test = static_dataset('validation','M',emotions,frame_number)
x_test,y_test,class_weight_dict_test = static_dataset('test','M',emotions,frame_number)


x = tf.keras.utils.normalize(x,    axis=-1,    order=2)

#search parameters for hyperparameter optimization
#batch_sizes = [16, 32, 64, 128, 256]
batch_sizes = [8]
#epochs = [10, 50, 100]
epochs = [300]
#optimizers = [sgd, rmsdrop, adagrad, adadelta, adam, adamax, nadam]
#learn_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2]
#learn_rates = [0.0001,0.00005,0.00001,0.000005,0.000001]
learn_rates = [0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000001,0.0000001]
#activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
activations = ['relu']
#dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_rates = [0.5]
#hidden_neurons = [[256,128,64,32],[128,64,32,16],[64,32,16,8],[256,256,256,256],[64,64,64,64],[32,32,32,32]]#, [500,256,128,64],[256,64,32,16], [256,128,64,32], [32,16,8,4]]
hidden_neurons = [[128,  64, 32, 16]]
#hidden2_neurons = [16, 32, 64, 128, 256]
#hidden3_neurons = [16, 32, 64, 128, 256]
#hidden4_neurons = [16, 32, 64, 128, 256]


for activation in activations:
	for dropout in dropout_rates:
		for hidden in hidden_neurons:
			model = Sequential()
			model.add(Dense(hidden[0], activation=activation, input_dim=frame_number*87, name='dense_1',kernel_initializer='glorot_uniform'))
			model.add(Dropout(dropout))
			model.add(Dense(hidden[1], activation=activation, name='dense_2',kernel_initializer='glorot_uniform'))
			model.add(Dropout(dropout))
			model.add(Dense(hidden[2], activation=activation, name='dense_3',kernel_initializer='glorot_uniform'))
			model.add(Dropout(dropout))
			model.add(Dense(hidden[3], activation=activation, name='dense_4',kernel_initializer='glorot_uniform'))

			model.add(Dense(len(emotions), activation='softmax',name='dense_f'))

			for learn_rate in learn_rates:
				#sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=False)
				#rmsdrop = keras.optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=None, decay=0.0)
				#adagrad = keras.optimizers.Adagrad(lr=learn_rate, epsilon=None, decay=0.0)
				#adadelta = keras.optimizers.Adadelta(lr=learn_rate, rho=0.95, epsilon=None, decay=0.0)
				adam =keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
				#adamax = keras.optimizers.Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
				#nadam = keras.optimizers.Nadam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

				#optimizers = [sgd, rmsdrop, adagrad, adadelta, adam, adamax, nadam]
				optimizers = [adam]

				#optimizer
				#adam =keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
				#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

				for batchSize in batch_sizes:
					for epoch in epochs:
						for optimizer in optimizers:

							model.compile(loss='categorical_crossentropy',
							              optimizer=optimizer,
							              metrics=['accuracy'])

							with open("code_validation_results.txt", 'a') as outputFile:
								outputFile.write("\n\n\nactivation= %s, dropout= %f, optimizer= %s, batch size = %d, epoch= %d, learning rate= %f" % (activation, dropout, optimizer, batchSize, epoch, learn_rate))
								outputFile.write("number of hidden1 neurons: %d, number of hidden2 neurons: %d, number of hidden3 neurons: %d, number of hidden4 neurons: %d" % (hidden[0], hidden[1], hidden[2], hidden[3]))
								#model.save_weights("weights")
								outputFile.write("\n   ---training---")
								outputFile.write(str(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0)))

							print(learn_rate)
							model.fit(x=x,y=y,batch_size=batchSize, epochs=epoch,shuffle=True,class_weight=class_weight_dict,validation_data=(x_val, y_val),callbacks=[plot_losses,earlystop])

							with open("code_validation_results.txt", 'a') as outputFile:
								outputFile.write("\n   ---training---")
								outputFile.write(str(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0)))

								outputFile.write("\n   ---test---  ")
								outputFile.write(str(numpy.sum(model.predict(x=x_test,batch_size=1)> 1/len(emotions),axis=0)))
								outputFile.write(str(model.evaluate(x=x_test,y=y_test,batch_size=1)))


