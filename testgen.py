#/usr/bin/python3
from os import system

mList = [30720, 15360, 7680, 64, 32, 16, 8, 4]
nList = [30720]
kList = [16]

for i in nList:
    for i2 in mList:
        system("./gen " + " -r " + str(i) + " -c " + str(i2) + " " + " test/a_" + str(i) + "_" + str(i2) + ".mtx")
        system("./gen -d " + " -r " + str(i) + " -c " + str(i2) + " " + " test/a_" + str(i) + "_" + str(i2) + ".d.mtx")
        for j in kList:
            system("./gen " + " -r " + str(i2) + " -c " + str(j) + " " + " test/b_" + str(i2) + "_" + str(j) + ".mtx")
            system("./gen -d " + " -r " + str(i2) + " -c " + str(j) + " " + " test/b_" + str(i2) + "_" + str(j) + ".d.mtx")

