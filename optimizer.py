import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from utils import dataset_generator
from utils import total_number

numpy.set_printoptions(threshold=numpy.inf)

trainable = 'True'


#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']

emotions = ['sad','ang', 'neu', 'exc']#,'ang','neu']
frame_number = 20
size_batch = 128


#some possible optimizer
learning_rate = 0.01
sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)
rmsdrop = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
adagrad = keras.optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
adadelta = keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
adam =keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adamax = keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
nadam = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


train_size,_,_ = total_number('train','M',emotions,size_batch,frame_number)
validation_size,_,_ = total_number('test','M',emotions,size_batch,frame_number)
test_size,_,_ = total_number('test','M',emotions,size_batch,frame_number)

print("\nsize of train "+str(train_size)+"\n")
print("size of test_size "+str(test_size)+"\n")

train_generator = dataset_generator(size_batch,'train','M',emotions,frame_number)
validation_generator = dataset_generator(size_batch,'validation','M',emotions,frame_number)
test_generator = dataset_generator(size_batch,'test','M',emotions,frame_number)

#search parameters for hyperparameter optimization
batch_sizes = [16, 32, 64, 128, 256]
epochs = [10, 50, 100]
#optimizers = [sgd, rmsdrop, adagrad, adadelta, adam, adamax, nadam]
learn_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2]
activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
hidden1_neurons = [16, 32, 64, 128, 256]
hidden2_neurons = [16, 32, 64, 128, 256]
hidden3_neurons = [16, 32, 64, 128, 256]
hidden4_neurons = [16, 32, 64, 128, 256]


for learn_rate in learn_rates:
	model = Sequential()
	model.add(Dense(256, activation='sigmoid', input_dim=frame_number*33, name='dense_1',kernel_initializer='VarianceScaling'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='sigmoid', name='dense_2',kernel_initializer='VarianceScaling'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='sigmoid', name='dense_3',kernel_initializer='VarianceScaling'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='sigmoid', name='dense_4',kernel_initializer='VarianceScaling'))
	model.add(Dense(len(emotions), activation='softmax',trainable=trainable,name='dense_55'))

	sgd = SGD(lr=learn_rate, decay=0, momentum=0.9, nesterov=False)

	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])

	train_size,_,_ = total_number('train','M',emotions,size_batch,frame_number)
	validation_size,_,_ = total_number('test','M',emotions,size_batch,frame_number)
	test_size,_,_ = total_number('test','M',emotions,size_batch,frame_number)

	print("\nsize of train "+str(train_size)+"\n")
	print("size of test_size "+str(test_size)+"\n")

	train_generator = dataset_generator(size_batch,'train','M',emotions,frame_number)
	validation_generator = dataset_generator(size_batch,'validation','M',emotions,frame_number)
	test_generator = dataset_generator(size_batch,'test','M',emotions,frame_number)



	model.fit_generator(train_generator, steps_per_epoch=train_size, epochs=1,shuffle=True,  use_multiprocessing =True, workers = 7 )

	pred = model.predict_generator( test_generator, steps=test_size)
	print(pred)
	print(numpy.sum(pred > 0.5,axis=0))

	print(model.evaluate_generator( test_generator, steps=test_size))
	print("activation= %s, dropout= %f, optimizer= %s, batch size = %d, epoch= %d, learning rate= %f" % (activation, dropout, optimizer, batchSize, epoch, learn_rate))
	print("number of hidden1 neurons: %d, number of hidden2 neurons: %d, number of hidden3 neurons: %d, number of hidden4 neurons: %d" % (hidden1, hidden2, hidden3, hidden4))
											




# for activation in activations:
# 	for dropout in dropout_rates:
# 		for hidden1 in hidden1_neurons:
# 			for hidden2 in hidden2_neurons:
# 				for hidden3 in hidden3_neurons:
# 					for hidden4 in hidden4_neurons:
# 						model = Sequential()
# 						model.add(Dense(hidden1, activation=activation, input_dim=frame_number*33,name='dense_1', kernel_initializer='VarianceScaling'))
# 						model.add(Dropout(dropout))
# 						model.add(Dense(hidden2, activation=activation, name='dense_2', kernel_initializer='VarianceScaling'))
# 						model.add(Dropout(dropout))
# 						model.add(Dense(hidden3, activation=activation, name='dense_3', kernel_initializer='VarianceScaling'))
# 						model.add(Dropout(dropout))
# 						model.add(Dense(hidden4, activation=activation, name='dense_4', kernel_initializer='VarianceScaling'))
# 						model.add(Dense(len(emotions), activation='softmax',trainable=trainable,name='dense_55'))

# 						for learn_rate in learn_rates:
# 							sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=False)
# 							rmsdrop = keras.optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=None, decay=0.0)
# 							adagrad = keras.optimizers.Adagrad(lr=learn_rate, epsilon=None, decay=0.0)
# 							adadelta = keras.optimizers.Adadelta(lr=learn_rate, rho=0.95, epsilon=None, decay=0.0)
# 							adam =keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# 							adamax = keras.optimizers.Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# 							nadam = keras.optimizers.Nadam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

# 							optimizers = [sgd, rmsdrop, adagrad, adadelta, adam, adamax, nadam]

# 							for batchSize in batch_sizes:
# 								for epoch in epochs:
# 									for optimizer in optimizers:
										
# 											model.compile(loss='categorical_crossentropy',
# 											              optimizer=optimizer,
# 											              metrics=['accuracy'])


# 											model.fit_generator(train_generator, steps_per_epoch=train_size, epochs=1,shuffle=True,  use_multiprocessing =True, workers = 7 )

# 											#model.load_weights('weights',by_name=False)

# 											#print(numpy.sum(model.predict_generator( train_generator, steps=test_size/batchSize ),axis=0))

# 											pred = model.predict_generator( test_generator, steps=test_size)
# 											print(pred)
# 											print(numpy.sum(pred > 0.5,axis=0))

# 											print(model.evaluate_generator( test_generator, steps=test_size))
# 											print("activation= %s, dropout= %f, optimizer= %s, batch size = %d, epoch= %d, learning rate= %f" % (activation, dropout, optimizer, batchSize, epoch, learn_rate))
# 											print("number of hidden1 neurons: %d, number of hidden2 neurons: %d, number of hidden3 neurons: %d, number of hidden4 neurons: %d" % (hidden1, hidden2, hidden3, hidden4))
# 											#model.save_weights("weights")




