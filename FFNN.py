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
from models import FFNN

print("library imported")


numpy.set_printoptions(threshold=numpy.inf)

#CALLBACK
plot_losses = PlotLosses()
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, \
                          verbose=1, mode='auto')


#SCRIPT PARAM
trainable = 'True'
n_different_training= 10
#epoc = 120 magic number
epoc = 120
feature_type = "HLF"
emotions = ['ang', 'exc', 'neu', 'sad']
#emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']


#TRANSFER LEARNING SETTINGS
trainable =True
load_weights = False
save_weight = False
wight_file_name = "null"

#MODEL PARAM
size_batch2 = 32
frame_number = 1
regu = 0.0000
bias = True
drop_rate = 0.2


feature_number=87
if feature_type =="LLF":
	feature_number=31



model = FFNN(trainable=trainable,feature_number=feature_number,frame_number=frame_number,emotions=emotions,lr=0.0001)



#FEMALE DATASET
x,y,class_weight_dict = static_dataset(feature_type,'train','F',emotions,frame_number)
x_v,y_v,_ = static_dataset(feature_type,'test','F',emotions,frame_number)

#FEMALE DATASET


print(len(x))
x = x[::4]
y = y[::4]





#DATASET NORMALIZATION
x = numpy.array(x)
x = tf.keras.utils.normalize(x,    axis=-1,    order=2)
y = numpy.array(y)


#cass_weight_dict = weight_class('train',emotions,'M')



avg_h_loss = []
avg_h_val_loss = []
avg_h_acc = []
avg_h_val_acc = []



#print("\n   ---training---")
#print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))

for i in range(0,n_different_training):
	print(i)

	history = model.fit(x=x,y=y,batch_size=size_batch2, epochs=epoc,shuffle=True,
				class_weight=class_weight_dict,validation_data=(x_v, y_v),callbacks=[plot_losses])

	print("\n   ---training---")
	print(numpy.sum(model.predict(x=x,batch_size=1)> 1/len(emotions),axis=0))
	print("\n   ---validation.. now test..---")
	print(numpy.sum(model.predict(x=x_v,batch_size=1)> 1/len(emotions),axis=0))

	reset_weights(model)



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


avg_h_loss = np.average(avg_h_loss,axis=0)
avg_h_val_loss = np.average(avg_h_val_loss,axis=0)
avg_h_acc = np.average(avg_h_acc,axis=0)
avg_h_val_acc = np.average(avg_h_val_acc,axis=0)








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





