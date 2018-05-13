import csv
import os, os.path

file_label = "data/data_label.csv"
file_feature_folder = 'data/IEMOCAP_feature'

with open(file_label, 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
	for row in spamreader:

		os.system('cp '+file_feature_folder+'/'+row[0]+'.csv'+' '+'data/'+row[1])
		print('cp '+file_feature_folder+'/'+row[0]+'.csv'+' '+'data/'+row[1])




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

 	
file.write(str(xxx_n)+";"+str(ang_n)+";"+str(dis_n)+";"+str(exc_n)+";"+str(fea_n)+
	";"+str(fru_n)+";"+str(hap_n)+";"+str(neu_n)+";"+str(sad_n)+";"+str(sur_n,)+";"+str(oth_n)+"\n")

file.write("xxx;ang;dis;exc;fea;fru;hap;neu;sad;sur;oth")

#file.write('xxx ; '+str(len([name for name in os.listdir(dir_xxx) if os.path.isfile(os.path.join(dir_xxx, name))]))+'\n') 
#file.write('ang ; '+str(len([name for name in os.listdir(dir_ang) if os.path.isfile(os.path.join(dir_ang, name))]))+'\n') 
#file.write('dis ; '+str(len([name for name in os.listdir(dir_dis) if os.path.isfile(os.path.join(dir_dis, name))]))+'\n') 
#file.write('exc ; '+str(len([name for name in os.listdir(dir_exc) if os.path.isfile(os.path.join(dir_exc, name))]))+'\n') 
#file.write('fea ; '+str(len([name for name in os.listdir(dir_fea) if os.path.isfile(os.path.join(dir_fea, name))]))+'\n') 
#file.write('fru ; '+str(len([name for name in os.listdir(dir_fru) if os.path.isfile(os.path.join(dir_fru, name))]))+'\n') 
#file.write('hap ; '+str(len([name for name in os.listdir(dir_hap) if os.path.isfile(os.path.join(dir_hap, name))]))+'\n') 
#file.write('neu ; '+str(len([name for name in os.listdir(dir_neu) if os.path.isfile(os.path.join(dir_neu, name))]))+'\n') 
#file.write('sad ; '+str(len([name for name in os.listdir(dir_sad) if os.path.isfile(os.path.join(dir_sad, name))]))+'\n') 
##file.write('sur ; '+str(len([name for name in os.listdir(dir_sur) if os.path.isfile(os.path.join(dir_sur, name))]))+'\n') 
##file.write('oth ; '+str(len([name for name in os.listdir(dir_oth) if os.path.isfile(os.path.join(dir_oth, name))]))+'\n') 

file.close() 


