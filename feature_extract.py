from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
[Fs, x] = audioBasicIO.readAudioFile("data/speechEmotion/00.wav");
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel('ZCR'); 
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel('Energy'); plt.show()





#python audioAnalysis.py featureExtractionFile -i data/speechEmotion/00.wav -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050 -o data/speechEmotion/00.wav




python pyAudioAnalysis/AaudioAnalysis.py featureExtractionFile -i data/speechEmotion/00.wav -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050 -o data/speechEmotion/00.wav


