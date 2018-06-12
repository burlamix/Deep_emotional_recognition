import csv
import numpy as np
import numpy
from keras import backend as K
from keras.layers import Dense
import keras

import os

ex = 'data/sad/Ses01F_impro02_F002.csv'
#ex = 'data/test.csv'

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if isinstance(layer, Dense):
            old = layer.get_weights()
            layer.kernel.initializer.run(session=session)
            layer.bias.initializer.run(session=session)
            print(np.array_equal(old, layer.get_weights())," after initializer run")
        else:
            print(layer, "not reinitialized")


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        #self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        #plt.plot(self.x, self.losses, label="loss")
        #plt.plot(self.x, self.val_losses, label="val_loss")
        #plt.legend()
        #plt.show();
        


def Categorical_label(label,emotion):
# define the function blocks

	hot_encoding = np.zeros(len(emotion))
	hot_encoding[emotion.index(label)] = 1


	return hot_encoding


def weight_class(file_name,emotion,gender):

	total = 0
	count = np.zeros(len(emotion))

	with open(os.getcwd()+"/data/"+file_name+"/batch_count_"+file_name, 'rt') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		spamreader = iter(spamreader)
		next(spamreader)

		for i in range(0,22):	
			row = next(spamreader)

			if (row[0][-1]==gender and any(row[0][:-2] in s for s in emotion) ):
				total = total + int(row[1])

				count[emotion.index(row[0][:-2])]=float(row[1])

	class_weight = [ (total-x) for x in count]
	min_n = min(count)
	class_weight = [ x/min_n for x in class_weight]

	class_weight_dict ={}

	for i in range(0,len(class_weight)):
		class_weight_dict.update({i:class_weight[i]})

	return class_weight_dict

def static_dataset(feature_type,folder,gender,emotion,frame_number,equal_size=False):

	data_generator = dataset_generator(1,feature_type,folder,gender,emotion,frame_number,stop=True)

	x=[]
	y=[]
	new_x=[]
	new_y=[]

	counterrr=0
	emo_counter_tot = np.zeros(len(emotion))
	emo_counter = np.zeros(len(emotion))

	total=0
	while True:
		total = total +1
		try:
			g = next(data_generator)
			#print(g[0][0])
			#exit()
			x.append(g[0][0])
			y.append(g[1][0])	

			emo_counter_tot[ np.nonzero(g[1][0])[0] ] = emo_counter_tot[np.nonzero(g[1][0])[0] ] + 1

		except StopIteration:
			break
	if equal_size == True:
		min_n = min(emo_counter_tot)
		total=0
		#making the samples weight		
		for i in range(0,len(x)):
	
			if(emo_counter[ np.nonzero(y[i])[0] ] <min_n):
				new_x.append(x[i])
				new_y.append(y[i])	
				emo_counter[ np.nonzero(y[i])[0] ] = emo_counter[ np.nonzero(y[i])[0] ] +1	
				total =total + 1
		emo_counter_tot = emo_counter
		x = new_x
		y = new_y

	class_weight = [ (total-x) for x in emo_counter_tot]
	min_n = min(class_weight)
	class_weight = [ x/min_n for x in class_weight]

	class_weight_dict ={}

	for i in range(0,len(class_weight)):
		class_weight_dict.update({i:class_weight[i]})

	print('finished data')
	return numpy.array(x),numpy.array(y),class_weight


def total_number(feature_type,file_name, gender, emotion,size_batch,frame_number):

	total = 0
	count = np.zeros(len(emotion))

	with open(os.getcwd()+"/data/"+feature_type+'/'+file_name+"/batch_count", 'rt') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		spamreader = iter(spamreader)
		next(spamreader)

		for i in range(0,22):	
			row = next(spamreader)

			if (row[0][-1]==gender and any(row[0][:-2] in s for s in emotion) ):
				total = total + int(row[1])

				count[emotion.index(row[0][:-2])]=float(row[1])

	prob = [x / total for x in count]

	total = ((total*5)/size_batch)/frame_number

	return total,emotion,prob


def safe_div(x,y):
	try: return float(x)/y
	except ZeroDivisionError: return 0

#reset probability of file in folder
def reset_probability(folder_name):

	new_probability=[]
	tot=0
	for folder in folder_name:
		file_number = len([name for name in os.listdir("data/"+folder) if os.path.isfile(os.path.join("data/"+folder, name))])
		tot = tot + file_number
		new_probability.append(file_number)
	new_probability = [safe_div(x, tot) for x in new_probability]

	return new_probability



#generator that return  five rows as array for each call
def from_file(file,emotion,frame_number):
	count = 0
	with open(file, 'rt') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		#skipp the first row
		spamreader = iter(spamreader)
		next(spamreader)

		mod=0
		for_to_pass =[]
		for row in spamreader:

			mod = mod +1

			#selecting part
			#selected_row = np.concatenate([ row[39:60] ,row[108:] ])
			selected_row = row[3:]


			#convert array of stringo to array of float skipping the first two element
			if frame_number == 1:
				for_to_pass.extend(list(map(float,selected_row)))
			else:
				for_to_pass.append(list(map(float,selected_row)))


			if mod==frame_number :

				yield	np.array(for_to_pass) , Categorical_label(row[0],emotion)
				break

		if mod < frame_number:
			while mod != frame_number:
				for_to_pass.append(list(map(float,selected_row)))
				mod = mod +1

			yield	np.array(for_to_pass) , Categorical_label(row[0],emotion)




#generator that return five rows as array for each call, from all the file in the fiven folder
def from_folder(folder,emotion,frame_number):
	#iterate on each file
	for subdir, dirs, files in os.walk(os.getcwd()+"/"+folder):
		for file in files:
			#create a generator to get all rows from a file
			iter_file = from_file(os.getcwd()+"/"+folder+"/"+file,emotion,frame_number)
			#iterate until the generator and and return a exception
			while True:
				try:
					new_d =np.array(next(iter_file))
					yield new_d
				except StopIteration:
					#end of the file
					break;



def dataset_generator(batch_size,feature_type,folder,gender,emotion,frame_number,stop=False):


	total,folder_name,probability = total_number(feature_type,folder, gender, emotion,batch_size,frame_number)

	initial_probability = probability

	generator_list = []
	no_stop = True
	#make a list of generator for each folder
	for name in folder_name:
		generator_list.append(from_folder("data/"+feature_type+'/'+folder+"/"+name+'_'+gender,emotion,frame_number))

	batch_counter = 0
	x_batch = []
	y_batch = []

	while (no_stop==True):
		#random with probability choise 
		x_folder = np.random.choice(numpy.arange(0, len(probability)), p=probability)

		try:
			# we don't handle the last not completely full batch ! in that case rememeber to remove batch counter = zero from the zero exception
				new_xy = np.array(next(generator_list[x_folder]))

				x_batch.append(new_xy[0])
				y_batch.append(new_xy[1])
				batch_counter = batch_counter+ 1

				#if the batch is full yield the batch, and start to make a new batch
				if batch_counter == batch_size :
					yield np.array(x_batch) , np.array(y_batch)
					batch_counter = 0
					x_batch = []
					y_batch = []

		#if one folder generator yiels an exception (it means that the folder is empty)
		except StopIteration:

			try:
				#reset the probabiltiy to don't use that folder
				probability = np.array(probability)
				non_zero = np.count_nonzero(probability) -1
				probability = np.where(probability, probability + (float(probability[x_folder])/non_zero), probability)
				probability[x_folder] = 0

			# if all folder are empty reset the generator and the probability
			except ZeroDivisionError:

				if (stop==True):
					no_stop = False

				probability = initial_probability
				generator_list = []
				for name in folder_name:
					generator_list.append(from_folder("data/"+folder+"/"+name+'_'+gender,emotion,frame_number))

				batch_counter = 0
				x_batch = []
				y_batch = []



#not used
def statistics(new_folder,gender,emotion=None):

	dir_ang = 'data/'+new_folder+'/ang_'+gender
	dir_dis = 'data/'+new_folder+'/dis_'+gender
	dir_exc = 'data/'+new_folder+'/exc_'+gender
	dir_fea = 'data/'+new_folder+'/fea_'+gender
	dir_fru = 'data/'+new_folder+'/fru_'+gender
	dir_hap = 'data/'+new_folder+'/hap_'+gender
	dir_neu = 'data/'+new_folder+'/neu_'+gender
	dir_oth = 'data/'+new_folder+'/oth_'+gender
	dir_sad = 'data/'+new_folder+'/sad_'+gender
	dir_sur = 'data/'+new_folder+'/sur_'+gender
	dir_xxx = 'data/'+new_folder+'/xxx_'+gender


	file = open('data/'+new_folder+'/statistics.txt','w') 

	file.write('statistics of the dataset\n')

	xxx_n = len([name for name in os.listdir(dir_xxx) if os.path.isfile(os.path.join(dir_xxx, name))])
	ang_n = len([name for name in os.listdir(dir_ang) if os.path.isfile(os.path.join(dir_ang, name))])
	dis_n = len([name for name in os.listdir(dir_dis) if os.path.isfile(os.path.join(dir_dis, name))])
	exc_n = len([name for name in os.listdir(dir_exc) if os.path.isfile(os.path.join(dir_exc, name))])
	fea_n = len([name for name in os.listdir(dir_fea) if os.path.isfile(os.path.join(dir_fea, name))])
	fru_n = len([name for name in os.listdir(dir_fru) if os.path.isfile(os.path.join(dir_fru, name))])
	hap_n = len([name for name in os.listdir(dir_hap) if os.path.isfile(os.path.join(dir_hap, name))])
	neu_n = len([name for name in os.listdir(dir_neu) if os.path.isfile(os.path.join(dir_neu, name))])
	sad_n = len([name for name in os.listdir(dir_sad) if os.path.isfile(os.path.join(dir_sad, name))])
	sur_n = len([name for name in os.listdir(dir_sur) if os.path.isfile(os.path.join(dir_sur, name))])
	oth_n = len([name for name in os.listdir(dir_oth) if os.path.isfile(os.path.join(dir_oth, name))])

	tot = xxx_n + ang_n + dis_n +exc_n+fea_n+fru_n+hap_n+neu_n+sad_n+sur_n+oth_n

	xxx_n = float(xxx_n)/tot
	ang_n = float(ang_n)/tot
	dis_n = float(dis_n)/tot
	exc_n = float(exc_n)/tot
	fea_n = float(fea_n)/tot
	fru_n = float(fru_n)/tot
	hap_n = float(hap_n)/tot
	neu_n = float(neu_n)/tot
	sad_n = float(sad_n)/tot
	sur_n = float(sur_n)/tot
	oth_n = float(oth_n)/tot

	prob = [xxx_n,ang_n,dis_n,exc_n,fea_n,fru_n,hap_n,neu_n,sad_n,sur_n,oth_n]
	name =  ["xxx","ang","dis","exc","fea","fru","hap","neu","sad","sur","oth"]

	file.write(str(xxx_n)+";"+str(ang_n)+";"+str(dis_n)+";"+str(exc_n)+";"+str(fea_n)+
		";"+str(fru_n)+";"+str(hap_n)+";"+str(neu_n)+";"+str(sad_n)+";"+str(sur_n)+";"+str(oth_n)+"\n")

	file.write("xxx;ang;dis;exc;fea;fru;hap;neu;sad;sur;oth")


	file.close() 

	return (prob,name)



#total_number('IEMOCAP_feature_test','M',['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx'])

#data_gen = dataset_generator(21,'train')


#for i in range (10000):
	#new_d =next(data_gen)
	#print(new_d[0].shape)#
	#print(new_d)#
	#print(next(data_gen))
	#print('\n\n\n')# rint 'The value of PI is approximately'
