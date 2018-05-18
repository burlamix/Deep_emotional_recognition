import csv
import numpy as np
import numpy

import os

ex = 'data/sad/Ses01F_impro02_F002.csv'
#ex = 'data/test.csv'

def Categorical_label(label):
# define the function blocks
	def ang():
	    return [1,0,0,0,0,0,0,0,0,0,0]
	def dis():
	    return [0,1,0,0,0,0,0,0,0,0,0]
	def exc():
	    return [0,0,1,0,0,0,0,0,0,0,0]
	def fea():
	    return [0,0,0,1,0,0,0,0,0,0,0]
	def fru():
	    return [0,0,0,0,1,0,0,0,0,0,0]
	def hap():
	    return [0,0,0,0,0,1,0,0,0,0,0]
	def neu():
	    return [0,0,0,0,0,0,1,0,0,0,0]
	def oth():
	    return [0,0,0,0,0,0,0,1,0,0,0]
	def sad():
	    return [0,0,0,0,0,0,0,0,1,0,0]
	def sur():
	    return [0,0,0,0,0,0,0,0,0,1,0]
	def xxx():
	    return [0,0,0,0,0,0,0,0,0,0,1]

	# map the inputs to the function blocks
	options = {'ang' : ang,
	           'dis' : dis,
	           'exc' : exc,
	           'fea' : fea,
	           'fru' : fru,
	           'hap' : hap,
	           'neu' : neu,
	           'oth' : oth,           
	           'sad' : sad,
	           'sur' : sur,
	           'xxx' : xxx,
	}
	return options[label]()




#generator that return  five rows as array for each call
def from_file(file):
	with open(file, 'rt') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		#skipp the first row
		spamreader = iter(spamreader)
		next(spamreader)

		mod=0
		five_on_row =[]
		for row in spamreader:
			mod = mod +1

			#convert array of stringo to array of float skipping the first two element
			five_on_row.extend(list(map(float,row[2:])))
			if mod==4 :

				#print(row[0])
				#print(Categorical_label(row[0]))
				yield	np.array(five_on_row) , Categorical_label(row[0])


				mod=0
				five_on_row =[]


#generator that return five rows as array for each call, from all the file in the fiven folder
def from_folder(folder):
	#iterate on each file
	for subdir, dirs, files in os.walk(os.getcwd()+"/"+folder):
		for file in files:
			#create a generator to get all rows from a file
			iter_file = from_file(os.getcwd()+"/"+folder+"/"+file)

			#iterate until the generator and and return a exception
			while True:
				try:
					new_d =np.array(next(iter_file))

					yield new_d
				except StopIteration:
					#end of the file
					break;


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




def dataset_generator(batch_size,segment):

	#get the number of file per folder
	with open(os.getcwd()+"/data/statistics.txt", 'rt') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
		spamreader = iter(spamreader)
		next(spamreader)
		probability = list(map(float,next(spamreader)))
		folder_name = next(spamreader)

	initial_probability = probability

	generator_list = []
	#make a list of generator for each folder
	for name in folder_name:
		generator_list.append(from_folder("data/"+segment+"/"+name))

	batch_counter = 0
	x_batch = []
	y_batch = []

	while True:
		#random with probability choise 
		x_folder = np.random.choice(numpy.arange(0, 11), p=probability)

		try:
			# we don't handle the last not completely full batch ! in that case rememeber to remove batch counter = zero from the zero exception
				new_xy = np.array(next(generator_list[x_folder]))

				x_batch.append(new_xy[0])
				y_batch.append(new_xy[1])
				batch_counter =batch_counter+ 1

				#if the batch is full yield the batch, and start to make a new batch
				if batch_counter == batch_size :
					yield np.array(x_batch) , np.array(y_batch)
					batch_counter = 0
					x_batch = []
					y_batch = []

		#if one folder generator yiels an excemption (it means that the folder is empty)
		except StopIteration:

			try:
				#reset the probabiltiy to don't use that folder
				probability = np.array(probability)
				non_zero = np.count_nonzero(probability) -1
				probability = np.where(probability, probability + (float(probability[x_folder])/non_zero), probability)
				probability[x_folder] = 0

			# if all folder are empty reset the generator and the probability
			except ZeroDivisionError:

				probability = initial_probability
				generator_list = []
				for name in folder_name:
					generator_list.append(from_folder("data/"+name))
				batch_counter = 0
				x_batch = []
				y_batch = []





#data_gen = dataset_generator(21,'train')


#for i in range (10000):
	#new_d =next(data_gen)
	#print(new_d[0].shape)#
	#print(new_d)#
	#print(next(data_gen))
	#print('\n\n\n')# rint 'The value of PI is approximately'
