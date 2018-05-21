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

emotions = ['sad','hap']#,'ang','neu']



model = Sequential()
model.add(Dense(256, activation='relu', input_dim=590, trainable=trainable,name='dense_1'))
#	model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', trainable=trainable,name='dense_2'))
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', trainable=trainable,name='dense_3'))
#model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', trainable=trainable,name='dense_4'))
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
sgd = SGD(lr=10, decay=1e-6, momentum=0.9, nesterov=False)


model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

train_size,_,_ = total_number('train','M',emotions)
validation_size,_,_ = total_number('test','M',emotions)
test_size,_,_ = total_number('test','M',emotions)

print("\nsize of train "+str(train_size)+"\n")
print("size of test_size "+str(test_size)+"\n")

train_generator = dataset_generator(64,'train','M',emotions)
validation_generator = dataset_generator(64,'validation','M',emotions)
test_generator = dataset_generator(64,'test','M',emotions)



model.fit_generator(validation_generator, steps_per_epoch=train_size/64, epochs=100,shuffle=True, use_multiprocessing =True, workers = 7 )

#model.load_weights('weights',by_name=False)
print(numpy.sum(model.predict_generator( train_generator, steps=test_size/64 ),axis=0))

print(model.evaluate_generator( test_generator, steps=test_size/64 ))

model.save_weights("weights")




