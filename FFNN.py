import keras
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from utils import dataset_generator
from utils import total_number
from utils import weight_class
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization



numpy.set_printoptions(threshold=numpy.inf)

#41 -- 51

trainable = 'True'
#6
numpy.random.seed(6)

#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']
emotions = ['hap','sad']#,'ang','exc']
size_batch = 32
frame_number = 20


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=frame_number*33, name='dense_1',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', name='dense_2',kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', name='dense_3',kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', name='dense_4',kernel_initializer='glorot_normal'))
#model.add(Dropout(0.5))
#model.add(Dense(100, activation='sigmoid', trainable=trainable,name='dense_5'))
#model.add(Dense(64, activation='sigmoid', trainable=trainable,name='dense_6'))
#model.add(Dense(32, activation='sigmoid', trainable=trainable,name='dense_7'))
#model.add(Dense(16, activation='sigmoid', trainable=trainable,name='dense_8'))
#model.add(Dense(8, activation='sigmoid', trainable=trainable,name='dense_9'))
#model.add(Dropout(0.5))
model.add(Dense(len(emotions), activation='softmax',name='dense_f'))


#some possible optimizer

adam =keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.0000005, decay=1e-6, momentum=0.9, nesterov=True)

#lr=0.0000001
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

train_size,_,_ = 		total_number('train','M',emotions,1,frame_number)
validation_size,_,_ = 	total_number('validation','M',emotions,size_batch,frame_number)
test_size,_,_ = 		total_number('train','M',emotions,size_batch,frame_number)


print("\nsize of train "+str(train_size)+"\n")
print("size of test_size "+str(test_size)+"\n")


train_generator = dataset_generator(1,'train','M',emotions,frame_number)
validation_generator = dataset_generator(size_batch,'validation','M',emotions,frame_number)
test_generator = dataset_generator(size_batch,'test','M',emotions,frame_number)

x=[]
y=[]
for i in range(0,int(train_size)):
	g = next(train_generator)
	x.append(g[0][0])
	y.append(g[1][0])

x = numpy.array(x)
x = tf.keras.utils.normalize(x,    axis=-1,    order=2)
y = numpy.array(y)

class_weight_dict = weight_class('train',emotions,'M')

#print(class_weight_dict)

#pred = model.predict_generator( validation_generator, steps=validation_size)
#print(numpy.sum(pred > 1/len(emotions),axis=0))
#print(model.evaluate_generator( validation_generator, steps=validation_size))

model.fit(x=x,y=y,batch_size=64, epochs=700,shuffle=True,class_weight=class_weight_dict)

#model.fit_generator(train_generator, steps_per_epoch=train_size, epochs=1500,shuffle=True,class_weight=class_weight_dict)


pred = model.predict_generator( validation_generator, steps=validation_size)
print(numpy.sum(pred > 1/len(emotions),axis=0))
print(model.evaluate_generator( validation_generator, steps=validation_size))


#model.load_weights('weights',by_name=False)
#model.save_weights("weights")




