import csv



ex = 'data/sad/Ses01F_impro02_F000.csv'


def insert_label_feature(file_to_change,label)
	new_file_rows =[]
	with open(file_to_change, 'r+') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
	     for row in spamreader:
	        print row
	        row[0] = label
	        new_file_rows.append(row)
	        print "\n\n"

	with open(file_to_change, 'wb') as file_to_write
		writer = csv.writer(file_to_write)
		writer.writerows(new_file_rows)
