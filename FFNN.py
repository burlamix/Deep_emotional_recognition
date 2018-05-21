import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from utils import dataset_generator
from utils import total_number
from sklearn.metrics import confusion_matrix

numpy.set_printoptions(threshold=numpy.inf)




trainable = 'True'


#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']

emotions = ['sad','ang']#,'ang','neu']
size_batch = 128
frame_number = 20

model = Sequential()
model.add(Dense(256, activation='sigmoid', input_dim=frame_number*33, name='dense_1',kernel_initializer='VarianceScaling'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='sigmoid', name='dense_2',kernel_initializer='VarianceScaling'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid', name='dense_3',kernel_initializer='VarianceScaling'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='sigmoid', name='dense_4',kernel_initializer='VarianceScaling'))
#model.add(Dropout(0.5))
#model.add(Dense(100, activation='sigmoid', trainable=trainable,name='dense_5'))
#model.add(Dense(64, activation='sigmoid', trainable=trainable,name='dense_6'))
#model.add(Dense(32, activation='sigmoid', trainable=trainable,name='dense_7'))
#model.add(Dense(16, activation='sigmoid', trainable=trainable,name='dense_8'))
#model.add(Dense(8, activation='sigmoid', trainable=trainable,name='dense_9'))
#model.add(Dropout(0.5))
model.add(Dense(len(emotions), activation='softmax',trainable=trainable,name='dense_55'))


#some possible optimizer
adam =keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.0005, decay=0, momentum=0.9, nesterov=False)

#lr=0.0000001
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



model.fit_generator(train_generator, steps_per_epoch=train_size, epochs=1,shuffle=True)

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




