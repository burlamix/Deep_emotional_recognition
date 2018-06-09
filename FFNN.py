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

from utils import dataset_generator
from utils import total_number
from utils import weight_class
from utils import static_dataset
from utils import PlotLosses
from utils import reset_weights

print("library imported")


numpy.set_printoptions(threshold=numpy.inf)

#CALLBACK
plot_losses = PlotLosses()
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, \
                          verbose=1, mode='auto')


#SCRIPT PARAM
trainable = 'True'
n_different_training= 70
epoc = 120
#emotions = ['ang', 'exc', 'neu', 'sad']
emotions = ['ang', 'exc', 'neu', 'sad']
#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']


#TRANSFER LEARNING SETTINGS
trainable =True
load_weights = False
save_weight = False


#MODEL PARAM
size_batch2 = 32
frame_number = 1
regu = 0.0000
bias = True
drop_rate = 0.2






model = Sequential()
model.add(Dense(254, activation='relu', input_dim=frame_number*87, name='dense_1',trainable=trainable,
	kernel_initializer='glorot_uniform',use_bias=bias,bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(regu,regu)))
model.add(Dropout(drop_rate))
model.add(Dense(254, activation='relu', name='dense_2',trainable=trainable,
	kernel_initializer='glorot_uniform',use_bias=bias,bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(regu,regu)))
model.add(Dropout(drop_rate))
model.add(Dense(254, activation='relu', name='dense_3',trainable=trainable,
	kernel_initializer='glorot_uniform',use_bias=bias,bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(regu,regu)))
model.add(Dropout(drop_rate))
model.add(Dense(254, activation='relu', name='dense_4',
	kernel_initializer='glorot_uniform',use_bias=bias,bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(regu,regu)))
#model.add(Dropout(drop_rate))


model.add(Dense(len(emotions), activation='softmax',name='dense_f'))


#optimizer
adadelta = keras.optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.0)
adam =keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.00000, decay=0, momentum=0, nesterov=True)

if load_weights == True:
	model.load_weights('weights',by_name=True)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])



#train_generator = dataset_generator(1,'train','M',emotions,frame_number,stop=True)
#validation_generator = dataset_generator(1,'validation','M',emotions,frame_number)
#test_generator = dataset_generator(1,'test','M',emotions,frame_number)


#resolving problem on unbalanced dataset
x,y,class_weight_dict = static_dataset('HLF','train','M',emotions,frame_number)
x_test,y_test,class_weight_dict_test = static_dataset('HLF','test','M',emotions,frame_number)
x_v,y_v,_ = static_dataset('HLF','validation','M',emotions,frame_number)


print("class weights")
print(class_weight_dict)

x = numpy.array(x)
x = tf.keras.utils.normalize(x,    axis=-1,    order=2)
y = numpy.array(y)

x_test = numpy.array(x_test)
y_test = numpy.array(y_test)

#cass_weight_dict = weight_class('train',emotions,'M')



avg_h_loss = []
avg_h_val_loss = []
avg_h_acc = []
avg_h_val_acc = []
avg_test_acc = []



#print("\n   ---training---")
#print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

for i in range(0,n_different_training):
	print(i)
	history = model.fit(x=x,y=y,batch_size=size_batch2, epochs=epoc,shuffle=True,
				class_weight=class_weight_dict,validation_data=(x_v, y_v),callbacks=[plot_losses])



	print("\n   ---training---")
	print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

	print("\n   ---validation---")
	print(numpy.sum(model.predict(x=x_v,batch_size=1)> 1/len(emotions),axis=0))

	print("\n   ---test---  ")
	print(numpy.sum(model.predict(x=x_test,batch_size=1)> 1/len(emotions),axis=0))
	acc = model.evaluate(x=x_test,y=y_test,batch_size=1)

	reset_weights(model)

	avg_test_acc.append(acc)
	print(acc)

	#STATISTIC
	avg_h_loss.append(history.history['loss'])
	avg_h_val_loss.append(history.history['val_loss'])
	avg_h_acc.append(history.history['acc'])
	avg_h_val_acc.append(history.history['val_acc'])




#COMPUTE STATISTICS
avg_h_loss = numpy.array(avg_h_loss)
avg_h_val_loss = numpy.array(avg_h_val_loss)
avg_h_acc = numpy.array(avg_h_acc)
avg_h_val_acc = numpy.array(avg_h_val_acc)
avg_test_acc = numpy.array(avg_test_acc)
avg_test_acc = np.average(avg_test_acc,axis=0)

print("avg---",str(avg_test_acc))

avg_h_loss = np.average(avg_h_loss,axis=0)
avg_h_val_loss = np.average(avg_h_val_loss,axis=0)
avg_h_acc = np.average(avg_h_acc,axis=0)
avg_h_val_acc = np.average(avg_h_val_acc,axis=0)


if load_weights == True:
	model.save_weights("weights")

#print("\n   ---training---")
#print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

#print("\n   ---test---  ")
#print(numpy.sum(model.predict(x=x_test,batch_size=1)> 1/len(emotions),axis=0))
#print(model.evaluate(x=x_test,y=y_test,batch_size=1))



#PRINT GRAPH
print(history.history.keys())
# summarize history for accuracy
plt.plot(avg_h_acc)
plt.plot(avg_h_val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(avg_h_loss)
plt.plot(avg_h_val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





