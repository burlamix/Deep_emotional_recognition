import csv
import numpy as np
import numpy

import os

ex = 'data/sad/Ses01F_impro02_F002.csv'
#ex = 'data/test.csv'



#generator that return array of feature form file
def from_file(file):
	with open(file, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	     spamreader = iter(spamreader)
	     next(spamreader)

	     mod=0
	     five_on_row =[]
	     for row in spamreader:
			
	        five_on_row.extend(list(map(float, row[2:])))
	        if mod==4 :

	        	yield	five_on_row , row[0]
	        	#yield row[0]
	         	mod=0
	     		five_on_row =[]
	        mod+=1

def from_folder(folder):

	for subdir, dirs, files in os.walk(os.getcwd()+"/"+folder):
		for file in files:

			iter_file = from_file(os.getcwd()+"/"+folder+"/"+file)

			while True:
			    try:
			        yield iter_file.next()
			    except StopIteration:
				  	#end of the file
			        break;


def safe_div(x,y):
    try: return float(x)/y
    except ZeroDivisionError: return 0

def reset_probability(folder_name):

	new_probability=[]
	tot=0
	for folder in folder_name:
		file_number = len([name for name in os.listdir("data/"+folder) if os.path.isfile(os.path.join("data/"+folder, name))])
		tot = tot + file_number
		new_probability.append(file_number)
	new_probability = [safe_div(x, tot) for x in new_probability]

	return new_probability




def dataset_generator(batch_size):

	#it's time to use some black magic:
	with open(os.getcwd()+"/data/statistics.txt", 'rb') as csvfile:
	    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
	    spamreader = iter(spamreader)
	    next(spamreader)
	    probability = list(map(float,next(spamreader)))
	    folder_name = next(spamreader)
	initial_probability = probability

	generator_list = []
	for name in folder_name:
		generator_list.append(from_folder("data/"+name))

	batch_counter = 0
	x_batch = []
	y_batch = []
	while  True:
		x_folder = np.random.choice(numpy.arange(0, 11), p=probability)


		try:
			# we don't handle the last not completely full batch ! in that case rememeber to remove batch counter = zero from the zero exception
				new_xy = generator_list[x_folder].next()
				x_batch.append(new_xy[0])
				y_batch.append(new_xy[1])
				batch_counter =batch_counter+ 1

				if batch_counter == 32 :
					yield np.array(x_batch) , np.array(y_batch)
					batch_counter = 0
					x_batch = []
					y_batch = []

		except StopIteration:

			try:
				probability = np.array(probability)
				non_zero = np.count_nonzero(probability) -1
				probability = np.where(probability, probability + (float(probability[x_folder])/non_zero), probability)
				probability[x_folder] = 0

			except ZeroDivisionError:

				probability = initial_probability
				generator_list = []
				for name in folder_name:
					generator_list.append(from_folder("data/"+name))
				batch_counter = 0
				x_batch = []
				y_batch = []





#data_gen = dataset_generator(32)

#for i in range (100000):

#	print len(data_gen.next()[0][130])
#	print data_gen.next()[0][130]
#	print "\n\n\n"



