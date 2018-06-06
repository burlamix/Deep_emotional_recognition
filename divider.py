import csv
import os, os.path



new_folder = "validation_g"

#file with the tuple "name file" "label"
file_label = "data/IEMOCAP_feature_"+new_folder+"/map_csv_label.csv"
#folder from where we have to take the data
file_feature_folder = 'data/IEMOCAP_feature_'+new_folder
#folder where write the file

os.system("mkdir -p "+new_folder)


with open(file_label, 'rt') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
	for row in spamreader:
		#print('mkdir -p data/'+new_folder+'/'+row[1]+'_'+row[0][-4])
		os.system('mkdir -p data/'+new_folder+'/'+row[1]+'_'+row[0][-4])
		os.system( " cp "+file_feature_folder+'/'+row[0]+'.csv ' +'data/'+new_folder+'/'+row[1]+'_'+row[0][-4])
		print( " cp "+file_feature_folder+'/'+row[0]+'.csv ' +'data/'+new_folder+'/'+row[1]+'_'+row[0][-4])






#also in some way we have to find the number of example and bath that we want fo cal fit_generator keras function....




