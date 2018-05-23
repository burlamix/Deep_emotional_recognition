import numpy as np
import csv
import os
import pickle
import h5py

sequenceLength = 50
emotions = ['ang','dis','exc','fea','fru','hap','neu','oth','sad','sur','xxx']
# print('1')aa
genders = ['M','F']
dataset_seperation = ['train', 'test', 'validation']
path =  os.getcwd() + '/data/'

for subset in dataset_seperation:
    subpath = path + subset
    print('Going to: '+ subpath)
    for gender in genders:
        for emotion in emotions:
            subsubpath = subpath + '/'+ emotion+'_'+gender + '/'
            print('Specificly '+ subsubpath)
            tmp = np.empty([999999, 33])
            counter = 0
            file_name = subset+'_'+emotion+'_'+gender
#             h5f = h5py.File(file_name+'.h5','r')
#             b = h5f[file_name][:]
#             h5f.close()
            if os.path.isfile('data/' +file_name+'.h5') == True:
                print('skipping, already done')    
                continue
#             print(os.path.isfile(file_name+'.h5'))
#             print('length is ' + b.shape)
                    
            for subdir, dirs, files in os.walk(subsubpath):
                # Files now contains all csv we need to import'
               
                for file in files:
                    with open(subsubpath+file, 'rt') as csvfile:
                        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                        pamreader = iter(spamreader)
                        next(spamreader)
                        for row in spamreader:
                            
                            selected_row = np.concatenate([ row[39:60] ,row[108:] ])
                            tmp[counter,:] = selected_row
                            counter+=1
                            if counter % 10000 == 1:
                                print(counter)
#                             if tmp == []:
#                                 tmp = selected_row
#                             else:
                                
#                             print(selected_row.shape)
#                             tmp.extend(list(map(float,selected_row)))
#                                 tmp = np.vstack((tmp, selected_row))
#                                 print(tmp.shape)
                            
            print('Finished with ' + emotion)
            tmp = tmp[1:counter,:]
            print(tmp.shape)
            h5f = h5py.File('data/'+file_name + '.h5', 'w')
            h5f.create_dataset(file_name, data=tmp)
            h5f.close()
            
#             pickle.dump( tmp, open(subset+'_'+emotion+'_'+gender+'.p', "wb" ) )

    
def Categorical_label(label,emotion):
# define the function blocks

    hot_encoding = np.zeros(len(emotion))
    hot_encoding[emotion.index(label)] = 1
    return hot_encoding

            