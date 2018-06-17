import sys
import os

# TODO: add file name of weights
weight_file = ""

#print help
if sys.argv[1] == "-h":
    print("First specify the path of the file to extract, then the path to the openSMILE installation.")

#read which file should be processed
if len(sys.argv) > 3:
    print("Too many command line arguments supplied! Only want the wav file name "
    "and the path to openSMILE.")
    sys.exit()

if len(sys.argv) <= 2:
    print("Too few command line arguments supplied! Want the wav file name "
    "and the path to openSMILE.")
    sys.exit()

audio_file_name = sys.argv[1]
opensmile_path = sys.argv[2]

#extract the feature vector to a file
os.system(opensmile_path + "SMILExtract -C "+ os.path.join(opensmile_path, "config/gemaps/eGeMAPSv01a.conf")
+ " -I "+ audio_file_name + " -D " + os.getcwd() + "emotion_analyzer_feature" + ".csv")

#load feature


#delete file again, we don't need it anymore


#define model
emotions = ['ang', 'exc', 'neu', 'sad']
model = FFNN(true,88,1,emotions,0.0001)

#load weights
model.load_weights(weight_file)

#predict


#print result
