import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
from utils import dataset_generator
from utils import total_number, weight_class
from sklearn.metrics import confusion_matrix
import h5py
import os
from utils import Categorical_label
import math


batch_size = 32
size_batch = batch_size
nb_feat = 33
nb_class = 4
nb_epoch = 80
emotions = ['sad','ang', 'neu', 'exc']#,'ang','neu']

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
                

# frame_number = 50
# # def build_simple_lstm(nb_feat, nb_class, 
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

# model.summary()

# # numpy.set_printoptions(threshold=numpy.inf)




# trainable = 'True'


# # train_size,_,_ = total_number('train','M',emotions,size_batch,frame_number)
# # validation_size,_,_ = total_number('test','M',emotions,size_batch,frame_number)
# # test_size,_,_ = total_number('test','M',emotions,size_batch,frame_number)

# # print("\nsize of train "+str(train_size)+"\n")
# # print("size of test_size "+str(test_size)+"\n")

# # train_generator = dataset_generator(size_batch,'train','M',emotions,frame_number, True)
# # validation_generator = dataset_generator(size_batch,'validation','M',emotions,frame_number)
# # test_generator = dataset_generator(size_batch,'test','M',emotions,frame_number)


# data_x, data_y = getData(['train'], emotions, 'M', 50)
# test_x, test_y = getData(['validation'], emotions, 'M', 50)
# hist  =  model.fit(data_x, data_y, batch_size=batch_size, epoch=nb_epoch, verbose=1, validation_data=(test_x, test_y))
# # model.fit_generator(train_generator, steps_per_epoch=train_size, epochs=250,shuffle=True,  use_multiprocessing = True, workers = 6 , class_weight=class_weight_dict)
# #model.load_weights('weights',by_name=False)
# pred = model.predict_generator( test_generator, steps=test_size)
# print(pred)
# print(np.sum(pred > 0.5,axis=0))

# #y_true =[]
# #test_generator = dataset_generator(128,'test','M',emotions)

# #for i in range (0,int(test_size/128)):
# #	y_true.append(next(test_generator)[1])
# 	#print(i)

# #y_true = np.array([0] * 1000 + [1] * 1000)
# #y_pred = numpy.sum(pred > 0.5,axis=0)

# #print(y_true)
# #print(len(y_true))
# #print(len(y_pred))

# #confusion_matrix(y_true, y_pred)

# print(model.evaluate_generator( test_generator, steps=test_size))

#model.save_weights("weights")




