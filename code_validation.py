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
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization



numpy.set_printoptions(threshold=numpy.inf)


trainable = 'True'


#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']
emotions = ['hap','sad','ang','exc']
size_batch2 = 32
frame_number = 50


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=frame_number*33, name='dense_1',kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', name='dense_2',kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', name='dense_3',kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', name='dense_4',kernel_initializer='glorot_normal'))

model.add(Dense(len(emotions), activation='softmax',name='dense_f'))


#optimizer
adam =keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])



x,y,class_weight_dict = static_dataset('train','M',emotions,frame_number)
x_test,y_test,class_weight_dict_test = static_dataset('test','M',emotions,frame_number)


x = tf.keras.utils.normalize(x,    axis=-1,    order=2)


print("\n   ---training---")
print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

model.fit(x=x,y=y,batch_size=size_batch2, epochs=300,shuffle=True,class_weight=class_weight_dict)

print("\n   ---training---")
print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

print("\n   ---test---  ")
print(numpy.sum(model.predict(x=x_test,batch_size=1)> 1/len(emotions),axis=0))
print(model.evaluate(x=x_test,y=y_test,batch_size=1))

