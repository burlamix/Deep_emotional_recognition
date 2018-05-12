import csv
import os, os.path

file_label = "data/data_label.csv"
file_feature_folder = 'data/IEMOCAP_feature'

#with open(file_label, 'rb') as csvfile:
#	spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
#	for row in spamreader:

#		os.system('cp '+file_feature_folder+'/'+row[0]+'.csv'+' '+'data/'+row[1])
		#print('cp '+file_feature_folder+'/'+row[0]+'.csv'+' '+'data/'+row[1])




#also in some way we have to find the number of example and bath that we want fo cal fit_generator keras function....

dir_ang = 'data/ang'
dir_dis = 'data/dis'
dir_exc = 'data/exc'
dir_fea = 'data/fea'
dir_fru = 'data/fru'
dir_hap = 'data/hap'
dir_neu = 'data/neu'
dir_oth = 'data/oth'
dir_sad = 'data/sad'
dir_sur = 'data/sur'
dir_xxx = 'data/xxx'


file = open('data/statistics.txt','w') 

file.write('statistics of the dataset')

file.write('ang ; '+str(len([name for name in os.listdir(dir_ang) if os.path.isfile(os.path.join(dir_ang, name))]))+'\n') 
file.write('dis ; '+str(len([name for name in os.listdir(dir_dis) if os.path.isfile(os.path.join(dir_dis, name))]))+'\n') 
file.write('exc ; '+str(len([name for name in os.listdir(dir_exc) if os.path.isfile(os.path.join(dir_exc, name))]))+'\n') 
file.write('fea ; '+str(len([name for name in os.listdir(dir_fea) if os.path.isfile(os.path.join(dir_fea, name))]))+'\n') 
file.write('fru ; '+str(len([name for name in os.listdir(dir_fru) if os.path.isfile(os.path.join(dir_fru, name))]))+'\n') 
file.write('hap ; '+str(len([name for name in os.listdir(dir_hap) if os.path.isfile(os.path.join(dir_hap, name))]))+'\n') 
file.write('neu ; '+str(len([name for name in os.listdir(dir_neu) if os.path.isfile(os.path.join(dir_neu, name))]))+'\n') 
file.write('oth ; '+str(len([name for name in os.listdir(dir_oth) if os.path.isfile(os.path.join(dir_oth, name))]))+'\n') 
file.write('sad ; '+str(len([name for name in os.listdir(dir_sad) if os.path.isfile(os.path.join(dir_sad, name))]))+'\n') 
file.write('sur ; '+str(len([name for name in os.listdir(dir_sur) if os.path.isfile(os.path.join(dir_sur, name))]))+'\n') 
file.write('xxx ; '+str(len([name for name in os.listdir(dir_xxx) if os.path.isfile(os.path.join(dir_xxx, name))]))+'\n') 

file.close() 


