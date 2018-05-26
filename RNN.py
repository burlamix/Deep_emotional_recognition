import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
from utils import dataset_generator
from utils import total_number, weight_class
import h5py
import os
from utils import Categorical_label
import math
# from RNN import getData
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint



batch_size = 128
nb_feat = 33
nb_class = 4
emotions = ['sad','ang', 'neu', 'exc']
frame_number = 50
def getData(sets, emotions, genders, frame_number): 
    data_y = [] 
    data_x = [] 
    for subset in sets: 
        for emotion in emotions: 
            for gender in genders: 
                file_name = subset+'_'+emotion+'_'+gender 
                h5f = h5py.File(os.getcwd()+'/data/'+file_name+'.h5','r') 
#                 print(list(h5f.keys())) 
                tmp = h5f[file_name][:] 
                h5f.close() 
                hot_encoded_emotion = Categorical_label(emotion, emotions) 
                a, b = tmp.shape 
#                 print(hot_encoded_emotion.shape) 
                if data_x == []: 
                    data_x = tmp 
#                     print(a) 
                    data_y = np.tile(hot_encoded_emotion,[a, 1]) 
                     
                else: 
                    data_x = np.vstack((tmp, data_x)) 
                    temp = np.tile(hot_encoded_emotion, [a, 1]) 
                    data_y = np.vstack((data_y, temp)) 
#                 print(data_x.shape) 
    samples = data_x.shape 
    newsize = math.trunc(samples[0]/frame_number) 
    x = np.reshape(data_x[0:newsize*frame_number,:], (newsize, frame_number, 33)) 
    # print(y[::20,:].shape) 
    y = data_y[::frame_number,:] 
    y = y[0:newsize,:] 
    return x, y 


def data():
	# maxlen = 100
	# max_features = 20000
	frame_number= 50
	emotions = ['sad','ang', 'neu', 'exc']
	print('Loading data...')
	X_train, y_train = getData(['train'], emotions, 'M', frame_number)
	X_test, y_test = getData(['test'], emotions, 'M', frame_number)
	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')

	print("Pad sequences (samples x time)")
	# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	return X_train, X_test, y_train, y_test

def model(X_train, X_test, y_train, y_test, batch_size, layer1, dropout, learn_rate):
	frame_number = 50
	nb_feat = 33
	nb_class = 4
	emotions = ['sad','ang', 'neu', 'exc']

	model = Sequential()
	# model.add(Embedding(max_features, 128, input_length=maxlen))
	model.add(LSTM(layer1, return_sequences = True, input_shape=(frame_number, nb_feat)))
	model.add(Activation('tanh'))
	model.add(LSTM(layer1, return_sequences = False))
	model.add(Activation('tanh'))
	model.add(Dense(256))
	model.add(Dropout(dropout))
	model.add(Activation('tanh'))
	model.add(Dense(nb_class))
	model.add(Activation('softmax'))
	class_weight_dict = weight_class('train',emotions,'M')
	myOptimizer = keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['accuracy'])


	early_stopping = EarlyStopping(monitor='val_loss', patience=4)
	checkpointer = ModelCheckpoint(filepath='results/Lr ' + str(learn_rate) + ' batchsize: ' + str(batch_size) + 'hidden '+ str(hidden1_neuron) + '.hdf5',
								   verbose=1,
								   save_best_only=True)



	model.fit(X_train, y_train,
			  batch_size=batch_size,
			  nb_epoch=5,
			  validation_split=0.08,
			  callbacks=[early_stopping, checkpointer])

	score, acc = model.evaluate(X_test, y_test, verbose=0)

	print('Test accuracy:', acc)

	return model, acc, score
X_train, X_test, y_train, y_test = data()
batch_sizes = [32, 64, 128, 256]
learn_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2]
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
hidden1_neurons = [128, 256, 512]
results = []
for learn_rate in learn_rates:
	for batch_size in batch_sizes:
		for dropout_rate in dropout_rates:
			for hidden1_neuron in hidden1_neurons:
				aa = 'Lr ' + str(learn_rate) + ' batchsize: ' + str(batch_size) + 'hidden '+ str(hidden1_neuron)
				fname = 'results/' + aa + '.hfd5'
				if os.path.isfile(fname) == False:
					print('file doesnt exist')
				else:
					print('file already exists, skipping')
					continue
				model, acc, score = model(X_train, X_test, y_train, y_test, batch_size, hidden1_neuron, dropout_rate, learn_rate)
				
				result.append(aa)
				print (aa)
