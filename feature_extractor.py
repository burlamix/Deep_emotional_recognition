import os

to_extract = "nope"
rootdir = '/Documents/univerista/delft/deep_learning/viv3/Emotional_recognition'
openS_path = "/home/simone/Documents/univerista/delft/deep_learning/viv3/opensmile-2.3.0/"
folder_for_feature = "/home/simone/Documents/univerista/delft/deep_learning/viv3/Emotional_recognition/data/IEMOCAP_feature_validation_g/"

count = 0

for subdir, dirs, files in os.walk(os.getcwd()+"/data/IEMOCAP_reduced/val"):
	for file in files:
		if file.endswith(".wav"):
		 	#print(os.path.join(subdir, file))
		 	to_extract = os.path.join(subdir, file)
		 	os.system(openS_path+"SMILExtract -C "+openS_path
		 			+"config/gemaps/eGeMAPSv01a.conf -I "+to_extract
		 			#+"config/gemaps/GeMAPSv01a.conf -I "+to_extract
		 			#+"config/emobase.conf -I "+to_extract
		 				+" -csvoutput "+folder_for_feature+os.path.splitext(file)[0]+".csv")
		 	count += 1
		 	

#GeMAPSv01a.conf

print("\n\n\n\tcorrectly extract feature from "+str(count)+" .wav files\n\n\n")


