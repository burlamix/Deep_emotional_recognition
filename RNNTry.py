import keras
import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
from utils import dataset_generator
from utils import total_number, weight_class
from sklearn.metrics import confusion_matrix


batch_size = 128	
size_batch = batch_size
nb_feat = 33
nb_class = 4
nb_epoch = 80
emotions = ['sad','ang', 'neu', 'exc']#,'ang','neu']

frame_number = 50


def data():
    maxlen = 100
    max_features = 20000

    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, X_test, y_train, y_test, max_features, maxlen

    
def model(X_train, X_test, y_train, y_test, max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM({{choice([128,256,512])}}, return_sequences = True, input_shape=(frame_number, nb_feat)))
    model.add(Activation('tanh'))
    model.add(LSTM({{choice([64, 128,256])}}, return_sequences = False))
    model.add(Activation('tanh'))
    model.add(Dense({{choice([128,256,512])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Activation('tanh'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
	class_weight_dict = weight_class('train',emotions,'M')


	model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)

    model.fit(X_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              nb_epoch=1,
              validation_split=0.08,
              callbacks=[early_stopping, checkpointer])

    score, acc = model.evaluate(X_test, y_test, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



# def build_simple_lstm(nb_feat, nb_class, 
# optimizer='Adadelta'
# optimizer =keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model = Sequential()
# model.add(LSTM(512, return_sequences=True, input_shape=(frame_number, nb_feat)))
# model.add(Activation('tanh'))
# model.add(LSTM(256, return_sequences=False))
# model.add(Activation('tanh'))
# model.add(Dense(512))
# model.add(Activation('tanh'))
# model.add(Dense(nb_class))
# model.add(Activation('softmax'))
# class_weight_dict = weight_class('train',emotions,'M')


# model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

    # return model

# def build_blstm(nb_feat, nb_class, optimizer='Adadelta'):
#     net_input = Input(shape=(78, nb_feat))
#     forward_lstm1  = LSTM(output_dim=64, 
#                           return_sequences=True, 
#                           activation="tanh"
#                          )(net_input)
#     backward_lstm1 = LSTM(output_dim=64, 
#                           return_sequences=True, 
#                           activation="tanh", 
#                           go_backwards=True
#                          )(net_input)
#     blstm_output1  = Merge(mode='concat')([forward_lstm1, backward_lstm1])
    
#     forward_lstm2  = LSTM(output_dim=64, 
#                           return_sequences=False, 
#                           activation="tanh"
#                          )(blstm_output1)
#     backward_lstm2 = LSTM(output_dim=64, 
#                           return_sequences=False, 
#                           activation="tanh", 
#                           go_backwards=True
#                          )(blstm_output1)
#     blstm_output2  = Merge(mode='concat')([forward_lstm2, backward_lstm2])
#     hidden = Dense(512, activation='tanh')(blstm_output2)
#     output = Dense(nb_class, activation='softmax')(hidden)
#     model  = Model(net_input, output)
    
#     model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
#     return model

# model = build_model()
model.summary()

numpy.set_printoptions(threshold=numpy.inf)




trainable = 'True'


#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']






# model = Sequential()
# model.add(Dense(256, activation='sigmoid', input_dim=frame_number*33, name='dense_1',kernel_initializer='VarianceScaling'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='sigmoid', name='dense_2',kernel_initializer='VarianceScaling'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='sigmoid', name='dense_3',kernel_initializer='VarianceScaling'))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='sigmoid', name='dense_4',kernel_initializer='VarianceScaling'))
#model.add(Dropout(0.5))
#model.add(Dense(100, activation='sigmoid', trainable=trainable,name='dense_5'))
#model.add(Dense(64, activation='sigmoid', trainable=trainable,name='dense_6'))
#model.add(Dense(32, activation='sigmoid', trainable=trainable,name='dense_7'))
#model.add(Dense(16, activation='sigmoid', trainable=trainable,name='dense_8'))
#model.add(Dense(8, activation='sigmoid', trainable=trainable,name='dense_9'))
#model.add(Dropout(0.5))
# model.add(Dense(len(emotions), activation='softmax',trainable=trainable,name='dense_55'))


#some possible optimizer
# adam =keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# sgd = SGD(lr=0.0005, decay=0, momentum=0.9, nesterov=False)

#lr=0.0000001
# # model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

train_size,_,_ = total_number('train','M',emotions,size_batch,frame_number)
validation_size,_,_ = total_number('test','M',emotions,size_batch,frame_number)
test_size,_,_ = total_number('test','M',emotions,size_batch,frame_number)

print("\nsize of train "+str(train_size)+"\n")
print("size of test_size "+str(test_size)+"\n")

train_generator = dataset_generator(size_batch,'train','M',emotions,frame_number, True)
validation_generator = dataset_generator(size_batch,'validation','M',emotions,frame_number)
test_generator = dataset_generator(size_batch,'test','M',emotions,frame_number)



model.fit_generator(train_generator, steps_per_epoch=train_size, epochs=250,shuffle=True,  use_multiprocessing = True, workers = 6 , class_weight=class_weight_dict)
#model.load_weights('weights',by_name=False)
pred = model.predict_generator( test_generator, steps=test_size)
print(pred)
print(numpy.sum(pred > 0.5,axis=0))

#y_true =[]
#test_generator = dataset_generator(128,'test','M',emotions)

#for i in range (0,int(test_size/128)):
#	y_true.append(next(test_generator)[1])
	#print(i)

#y_true = np.array([0] * 1000 + [1] * 1000)
#y_pred = numpy.sum(pred > 0.5,axis=0)

#print(y_true)
#print(len(y_true))
#print(len(y_pred))

#confusion_matrix(y_true, y_pred)

print(model.evaluate_generator( test_generator, steps=test_size))

#model.save_weights("weights")




