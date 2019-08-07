#/usr/bin/python3
from os import system

nList = [10240, 15360, 20480, 25600, 30720]
kList = [2, 4, 6, 8]

for i in nList:
    system("./gen " + " -r " + str(i) + " -c " + str(i) + " " + " test/a_" + str(i) + "_" + str(i) + ".mtx")
    system("./gen -d " + " -r " + str(i) + " -c " + str(i) + " " + " test/a_" + str(i) + "_" + str(i) + ".d.mtx")
    for j in kList:
        system("./gen " + " -r " + str(i) + " -c " + str(j) + " " + " test/b_" + str(i) + "_" + str(j) + ".mtx")
        system("./gen -d " + " -r " + str(i) + " -c " + str(j) + " " + " test/b_" + str(i) + "_" + str(j) + ".d.mtx")

