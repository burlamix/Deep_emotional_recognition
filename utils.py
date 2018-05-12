import csv



ex = 'data/sad/Ses01F_impro02_F001.csv'
#ex = 'data/test.csv'




def firstn(file):
	print file
	with open(file, 'rb') as csvfile:
	     spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
	     spamreader = iter(spamreader)
	     next(spamreader)
	     for row in spamreader:
	         yield  list(map(float, row[1:]))
	         #print "\n\n"

sum_of_first_n = firstn(ex)


print sum_of_first_n.next()	
print sum_of_first_n.next()[2]
print sum_of_first_n.next()[0]
print sum_of_first_n.next()[0]
print sum_of_first_n.next()[0]
