import random
import pandas
import os

inputlist = "kinetics_train_full.csv"
outlist ='kinetics_train_full_shuffle.csv'
rep = 100

flist = []
with open(inputlist, 'r') as f:
    header = next(f)
    for line in f:
        flist.append(line)

print("input list: {}".format(inputlist))
print("number of data before shuffle: {}".format(len(flist)))

f2 = open(outlist, 'w')
f2.write(header)
for t in range(100):
    listlen = len(flist)
    indices = range(listlen)
    random.shuffle(indices)

    for i in range(listlen):
        nowid = indices[i]
        line = flist[nowid]
        f2.write(line)
f2.close()

rep_list = pandas.read_csv(outlist)
print("output list: {}".format(outlist))
print("number of data with shuffling for {} times: {}".format(rep, rep_list.shape[0]))

