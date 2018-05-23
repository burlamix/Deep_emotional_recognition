from utils import get_samples, dataset_generator
import numpy as np
emotions = ['sad','hap']
# x, y = get_samples('train','M', emotions)

train_generator = dataset_generator(1,'train','M',emotions,50)
x_train = []
y_train = []
for x,y in train_generator:
	reshaped = x.reshape([50,33])
	x_train.append(reshaped)
	print(len(x_train))
	y_train.append(y)
