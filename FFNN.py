import keras
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from utils import dataset_generator
from utils import total_number
from utils import weight_class
from utils import static_dataset
from utils import PlotLosses

print("library imported")

plot_losses = PlotLosses()


numpy.set_printoptions(threshold=numpy.inf)

#41 -- 51

trainable = 'True'
#6
 #numpy.random.seed(6)

#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']
emotions = ['hap','sad']#,'ang','exc']
size_batch2 = 32
frame_number = 50


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=frame_number*33, name='dense_1',kernel_initializer='glorot_uniform'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', name='dense_2',kernel_initializer='glorot_uniform'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', name='dense_3',kernel_initializer='glorot_uniform'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', name='dense_4',kernel_initializer='glorot_uniform'))
#model.add(Dropout(0.5))
#model.add(Dense(100, activation='sigmoid', trainable=trainable,name='dense_5'))
#model.add(Dense(64, activation='sigmoid', trainable=trainable,name='dense_6'))
#model.add(Dense(32, activation='sigmoid', trainable=trainable,name='dense_7'))
#model.add(Dense(16, activation='sigmoid', trainable=trainable,name='dense_8'))
#model.add(Dense(8, activation='sigmoid', trainable=trainable,name='dense_9'))
#model.add(Dropout(0.5))
model.add(Dense(len(emotions), activation='softmax',name='dense_f'))


#optimizer
adam =keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

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

model.fit(x=x,y=y,batch_size=size_batch2, epochs=500,shuffle=True,class_weight=class_weight_dict,validation_data=(x_v, y_v),callbacks=[plot_losses])

print("\n   ---training---")
print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

print("\n   ---test---  ")
print(numpy.sum(model.predict(x=x_test,batch_size=1)> 1/len(emotions),axis=0))
print(model.evaluate(x=x_test,y=y_test,batch_size=1))

#model.fit_generator(train_generator, steps_per_epoch=train_size, epochs=1500,shuffle=True,class_weight=class_weight_dict)

#trained prediction
#pred = model.predict_generator( validation_generator, steps=validation_size)
#print(numpy.sum(pred > 1/len(emotions),axis=0))
#print(model.evaluate_generator( validation_generator, steps=validation_size))


#model.load_weights('weights',by_name=False)
#model.save_weights("weights")




