import os

to_extract = "nope"
rootdir = '/Documents/univerista/delft/deep_learning/venv/Emotional_recognition'
openS_path = "/home/simone/Documents/univerista/delft/deep_learning/venv/openSMILE-2.1.0/"
folder_for_feature = "/home/simone/Documents/univerista/delft/deep_learning/venv/Emotional_recognition/data/IEMOCAP_feature/"

count = 0

for subdir, dirs, files in os.walk(os.getcwd()+"/data/IEMOCAP_reduced/session1/audio/Ses01F_impro01"):
	for file in files:
		if file.endswith(".wav"):
		 	#print(os.path.join(subdir, file))
		 	to_extract = os.path.join(subdir, file)
		 	os.system(openS_path+"SMILExtract -C "+openS_path
		 			+"my_config/IS11_speaker_state.conf -I "+to_extract
		 				+" -D "+folder_for_feature+os.path.splitext(file)[0]+".csv")
		 	count += 1



print("\n\n\n\tcorrectly extract feature from "+str(count)+" .wav files\n\n\n")


