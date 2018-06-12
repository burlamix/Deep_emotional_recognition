import os

to_extract = "nope"
rootdir = '/Documents/univerista/delft/deep_learning/viv3/Emotional_recognition'
openS_path = "/home/simone/Documents/univerista/delft/deep_learning/viv3/opensmile-2.3.0/"
folder_for_feature = "/home/simone/Documents/univerista/delft/deep_learning/viv3/Emotional_recognition/data/LLF/val_feature/"

folder_from_data = os.walk(os.getcwd()+"/data/IEMOCAP_reduced/val")
count = 0

for subdir, dirs, files in folder_from_data:
	for file in files:
		if file.endswith(".wav"):
		 	#print(os.path.join(subdir, file))
		 	to_extract = os.path.join(subdir, file)
		 	os.system(openS_path+"SMILExtract -C "+openS_path
		 			#+"config/gemaps/eGeMAPSv01a.conf -I "+to_extract
		 			#+"config/gemaps/GeMAPSv01a.conf -I "+to_extract
		 			+"config/IS09_emotion.conf -I "+to_extract
		 			+" -D "+folder_for_feature+os.path.splitext(file)[0]+".csv"
		 			#+"  ../-frameModeFunctionalsConf shared/FrameModeFunctionals.conf.inc "
		 			)
		 	count += 1
		 	

#GeMAPSv01a.conf

print("\n\n\n\tcorrectly extract feature from "+str(count)+" .wav files\n\n\n")


