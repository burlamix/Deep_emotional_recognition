import os
import csv

input_folder = "/data/IEMOCAP_reduced/session1"
output_folder = "/data/IEMOCAP_feature"
file_output = "data/data_label.csv"

count = 0

def insert_label_feature(file_to_change,label):
	new_file_rows =[]
	with open(file_to_change, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
	     spamreader = iter(spamreader)
	     new_file_rows.append(next(spamreader))
	     for row in spamreader:
	        print row
	        row[0] = label
	        new_file_rows.append(row)
	        print "\n\n"

	with open(file_to_change, 'wb') as file_to_write:
		writer = csv.writer(file_to_write)
		writer.writerows(new_file_rows)


with open(file_output, 'wb') as csvfile:

	for subdir, dirs, files in os.walk(os.getcwd()+input_folder):
		for file in files:
			if file.endswith(".txt"):
			 	print(os.path.join(subdir, file))

				with open(os.path.join(subdir, file),'r') as f:
					spamwriter = csv.writer(csvfile)
					for line in f:
						interest_row = "["
						if line[0] == interest_row:
							line_splitted = line.split()
							#print(line)
							print(line_splitted[3]+" "+line_splitted[4])
							spamwriter.writerow([line_splitted[3],line_splitted[4]])

							insert_label_feature(os.getcwd()+"/data/IEMOCAP_feature/"+line_splitted[3]+".csv",line_splitted[4])

			 				count += 1

		 	#break;

print("\n\n\n\tcorrectly parsed label of "+str(count)+"  files\n\n\n")



'''


'''