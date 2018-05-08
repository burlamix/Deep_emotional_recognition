import os
import csv

file_name = "Ses01F_impro01.txt"
file_output = "data/data_label.csv"

count = 0
with open(file_output, 'wb') as csvfile:

	for subdir, dirs, files in os.walk(os.getcwd()+"/IEMOCAP_reduced/session1"):
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

			 				count += 1

		 	#break;

print("\n\n\n\tcorrectly parsed label of "+str(count)+"  files\n\n\n")



'''


'''