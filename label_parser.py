import os
import csv
import re

#the paths assume: all feature CSSV files are in base_path, all label files in
#base_path + labeltxt_sub_path
base_path = os.getcwd()+"/data/IEMOCAP_feature_validation_g/"
#base_path = "/Users/niklasbachmaier/IEMOCAP_Data/dummy_test/"
labeltxt_sub_path = "emo_evaluation"
map_csv_label_file = "map_csv_label"
mod5_file = "batch_count"

file_count = 0

#regex to check if sentence recorded from male or female
m_regex = re.compile('_M')

#variables to capture statistics

count_mod5 = {'ang_M': 0, 'dis_M': 0, 'exc_M': 0, 'fea_M': 0, 'fru_M': 0, 'hap_M': 0, 'neu_M': 0, 'oth_M': 0, 'sad_M': 0, 'sur_M': 0, 'xxx_M': 0, 'ang_F': 0, 'dis_F': 0, 'exc_F': 0, 'fea_F': 0, 'fru_F': 0, 'hap_F': 0, 'neu_F': 0, 'oth_F': 0, 'sad_F': 0, 'sur_F': 0, 'xxx_F': 0}

def insert_label_feature(file_to_change,label):
    new_file_rows =[]

    #determine which counter variable to count up (determined from label and male/female info)
    if m_regex.search(file_to_change):
        count_var = label + '_M'
    else:
        count_var = label + '_F'

    with open(file_to_change,'r') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
         spamreader = iter(spamreader)
         new_file_rows.append(next(spamreader))
         line_counter = 0
         for row in spamreader:
            line_counter += 1
            row[0] = label
            new_file_rows.append(row)
            count_mod5[count_var] += 1

    with open(file_to_change,'w') as file_to_write:
        writer = csv.writer(file_to_write)
        writer.writerows(new_file_rows)

    return line_counter


with open(base_path + map_csv_label_file,'w') as mapfile:
    print(base_path+labeltxt_sub_path)
    spamwriter = csv.writer(mapfile)
    for path, dirs, files in os.walk(base_path+labeltxt_sub_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(path, file)) as f:
                    for line in f:
                        interest_row = "["
                        if line[0] == interest_row:
                            line_splitted = line.split()

                            print(base_path+line_splitted[3]+".csv")
                            total_lines = insert_label_feature(base_path+line_splitted[3]+".csv",line_splitted[4])

                            spamwriter.writerow([line_splitted[3],line_splitted[4],total_lines])

                            file_count += 1

             #break;

print("\ncorrectly parsed labels of "+str(file_count)+"  files\n")

with open(base_path + mod5_file,'w') as f:
    print("mod5 statistics:")
    spamwriter = csv.writer(f)
    spamwriter.writerow(["Number of 5-frame-batches of each emotion, differentiated by gender"])
    for count_var in count_mod5:
        #print for overview on terminal and write to file for later use
        print (count_var, count_mod5[count_var])
        spamwriter.writerow([count_var,count_mod5[count_var]])
'''


'''
