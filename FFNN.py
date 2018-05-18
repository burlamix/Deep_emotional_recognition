import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from utils import dataset_generator

trainable = 'True'

numpy.random.seed(5)

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_dim=472, trainable=trainable,name='dense_1'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid', trainable=trainable,name='dense_2'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid', trainable=trainable,name='dense_3'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid', trainable=trainable,name='dense_4'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax',name='dense_55'))


sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#Make sure that there is a data\train folder and a data\validation folder containing all CSV's!
training_generator = dataset_generator(16, 'train', 5)
validation_generator = dataset_generator(16, 'validation', 5) 

model.fit_generator(generator = training_generator, validation_data = validation_generator,
                    steps_per_epoch=10, epochs=1000)

#model.load_weights('weights',by_name=False)

#print( model.predict_generator( dataset_generator(16,"test","male"), steps=1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0))

#model.save_weights("weights")




