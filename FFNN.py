import keras
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from utils import dataset_generator
from utils import total_number
from utils import weight_class
from utils import static_dataset
from utils import PlotLosses
print("library imported")

plot_losses = PlotLosses()
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=100, \
                          verbose=1, mode='auto')

numpy.set_printoptions(threshold=numpy.inf)


trainable = 'True'


#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']
emotions = ['hap','sad','ang','exc']
size_batch2 = 32
frame_number = 1
regu = 0.0000
bias = True
epoc = 600
drop_rate = 0.01

#TRANSFER LEARNING SETTINGS
trainable =True
load_weights = False
save_weight = False

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

# 51 --lr=0.000008, d =0

#optimizer
adam =keras.optimizers.Adam(lr=0.000008, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.00000, decay=0, momentum=0, nesterov=True)

if load_weights == True:
	model.load_weights('weights',by_name=True)


model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])



#train_size,_,_ = 		total_number('train','M',emotions,1,frame_number)
#validation_size,_,_ = 	total_number('validation','M',emotions,1,frame_number)
#test_size,_,_ = 		total_number('test','M',emotions,1,frame_number)


#print("\nsize of train "+str(train_size)+"\n")
#print("size of test_size "+str(test_size)+"\n")


#train_generator = dataset_generator(1,'train','M',emotions,frame_number,stop=True)
#validation_generator = dataset_generator(1,'validation','M',emotions,frame_number)
#test_generator = dataset_generator(1,'test','M',emotions,frame_number)



x,y,class_weight_dict = static_dataset('train','M',emotions,frame_number)
#random
#x = numpy.random.random((int(1312), frame_number*33))


x_test,y_test,class_weight_dict_test = static_dataset('test','M',emotions,frame_number)

x_v,y_v,_ = static_dataset('validation','M',emotions,frame_number)

print("class weights")
print(class_weight_dict)

x = numpy.array(x)
#x = numpy.random.random((int(train_size), frame_number*33))
x = tf.keras.utils.normalize(x,    axis=-1,    order=2)
y = numpy.array(y)

x_test = numpy.array(x_test)
y_test = numpy.array(y_test)

#cass_weight_dict = weight_class('train',emotions,'M')


#random initial prediciton
#pred = model.predict_generator( validation_generator, steps=validation_size)
#print(numpy.sum(pred > 1/len(emotions),axis=0))
#print(model.evaluate_generator( validation_generator, steps=validation_size))
print("\n   ---training---")
print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

history = model.fit(x=x,y=y,batch_size=size_batch2, epochs=epoc,shuffle=True,class_weight=class_weight_dict,validation_data=(x_v, y_v),callbacks=[earlystop,plot_losses])

if load_weights == True:
	model.save_weights("weights")

print("\n   ---training---")
print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

print("\n   ---test---  ")
print(numpy.sum(model.predict(x=x_test,batch_size=1)> 1/len(emotions),axis=0))
print(model.evaluate(x=x_test,y=y_test,batch_size=1))



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





