import csv
import os

file_label = "data/data_label.csv"
file_feature_folder = 'data/IEMOCAP_feature'

with open(file_label, 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
	for row in spamreader:

		os.system('cp '+file_feature_folder+'/'+row[0]+'.csv'+' '+'data/'+row[1])
		#print('cp '+file_feature_folder+'/'+row[0]+'.csv'+' '+'data/'+row[1])


