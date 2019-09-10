# Brute force parameter optimization for TSM2
# by Cody Rivera


# you will need to modify your subprocess.check_output calls to use this script
# in other installations

import subprocess
import csv
from os import system


# True Raw Optimal Parameters - Not adjusted

optParamsSingle = {
    (10240, 2)  : (2, 44),
    (10240, 4)  : (4, 44),
    (10240, 6)  : (6, 32),
    (10240, 8)  : (8, 24),
    (10240, 16) : (16, 16),
    (15360, 2)  : (2, 32),
    (15360, 4)  : (4, 32),
    (15360, 6)  : (6, 32),
    (15360, 8)  : (8, 68),
    (15360, 16) : (16, 32),
    (20480, 2)  : (2, 28),
    (20480, 4)  : (4, 32),
    (20480, 6)  : (6, 32),
    (20480, 8)  : (8, 24),
    (20480, 16) : (16, 32),
    (25600, 2)  : (2, 20),
    (25600, 4)  : (4, 32),
    (25600, 6)  : (6, 32),
    (25600, 8)  : (8, 16),
    (25600, 16) : (16, 12),
    (30720, 2)  : (2, 24),
    (30720, 4)  : (4, 12),
    (30720, 6)  : (6, 12),
    (30720, 8)  : (8, 12),
    (30720, 16) : (16, 20)
}

optParamsDouble = {
    (10240, 2)  : (2, 36),
    (10240, 4)  : (4, 24),
    (10240, 6)  : (6, 16),
    (10240, 8)  : (8, 20),
    (10240, 16) : (16, 16),
    (15360, 2)  : (2, 28),
    (15360, 4)  : (4, 24),
    (15360, 6)  : (6, 16),
    (15360, 8)  : (8, 20),
    (15360, 16) : (16, 12),
    (20480, 2)  : (2, 12),
    (20480, 4)  : (4, 12),
    (20480, 6)  : (6, 16),
    (20480, 8)  : (8, 12),
    (20480, 16) : (16, 12),
    (25600, 2)  : (2, 32),
    (25600, 4)  : (4, 32),
    (25600, 6)  : (6, 16),
    (25600, 8)  : (8, 12),
    (25600, 16) : (16, 12),
    (30720, 2)  : (2, 16),
    (30720, 4)  : (4, 32),
    (30720, 6)  : (6, 16),
    (30720, 8)  : (8, 16),
    (30720, 16) : (16, 12)
}

t1Value = 128



# Writes CSV header

def writeCSVHeader(wo):
    wo.writerow(["Precision", "n", "m", "k", "CUBLAS GFLOPS", "TSM2 GFLOPS", "Speedup"])

# Writes new parameter file

def writeParameterFile(t1, t2, t3):
    f = open("parameters.cuh", 'w')
    f.write("#define SINGLE_PARAM true" + "\n")
    f.write("#define FLOAT_T1 " + str(t1) + "\n")
    f.write("#define FLOAT_T2 " + str(t2) + "\n")
    f.write("#define FLOAT_T3 " + str(t3) + "\n")
    f.write("#define DOUBLE_T1 " + str(t1) + "\n")
    f.write("#define DOUBLE_T2 " + str(t2) + "\n")
    f.write("#define DOUBLE_T3 " + str(t3) + "\n")
    f.close()

def writeRegularParameterFile():
    f = open("parameters.cuh", 'w')
    f.write("//#define SINGLE_PARAM true" + "\n")
    f.write("#define FLOAT_T1 " + str(t1Value) + "\n")
    f.write("#define FLOAT_T2 " + str(32) + "\n")
    f.write("#define FLOAT_T3 " + str(32) + "\n")
    f.write("#define DOUBLE_T1 " + str(t1Value) + "\n")
    f.write("#define DOUBLE_T2 " + str(32) + "\n")
    f.write("#define DOUBLE_T3 " + str(32) + "\n")
    f.close()



# Compiles the program
def compileProgram(t2, t3):
    writeParameterFile(t1Value, t2, t3)
    subprocess.check_output(["make"], stderr=subprocess.DEVNULL)

def compileRegularProgram():
    writeRegularParameterFile()
    subprocess.check_output(["make"], stderr=subprocess.DEVNULL)


# Testbench function

def testSingleValues(n, m, k):
    try:
        single = subprocess.check_output(["ssh", "gpu01", "cd re*/ts* ; ./multiply test/a_" + str(n) + "_" + str(m) + ".mtx "
                                          + "test/b_" + str(m) + "_" + str(k) + ".mtx out"])
        
        singleCVal, singleVal = 0, 0
        try:
            # Horrendously hacky
            singleCVal = float(single.decode('utf-8').splitlines()[2].split(": ")[1].split(" G")[0])
            singleVal = float(single.decode('utf-8').splitlines()[3].split(": ")[1].split(" G")[0])
        except:
            singleCVal, singleVal = 0, 0
    
        return singleCVal, singleVal
    except:
        genAMatrix(n, m)
        genBMatrix(m, k)
        return testSingleValues(n, m, k)


def testDoubleValues(n, m, k):
    try:
        double = subprocess.check_output(["ssh", "gpu01", "cd re*/ts* ; ./multiply -d test/a_" + str(n) + "_" + str(m) + ".d.mtx "
                                          + "test/b_" + str(m) + "_" + str(k) + ".d.mtx out"])

        doubleCVal, doubleVal = 0, 0
        
        try:
            # Horrendously hacky
            doubleCVal = float(double.decode('utf-8').splitlines()[2].split(": ")[1].split(" G")[0])
            doubleVal = float(double.decode('utf-8').splitlines()[3].split(": ")[1].split(" G")[0])
        except:
            doubleCVal, doubleVal = 0, 0
    
        return doubleCVal, doubleVal
    except:
        genAMatrix(n, m)
        genBMatrix(m, k)
        return testDoubleValues(n, m, k)



# Matrix generation
def genAMatrix(a, b):
    system("./gen " + " -r " + str(a) + " -c " + str(b) + " " + " test/a_" + str(a) + "_" + str(b) + ".mtx")
    system("./gen -d " + " -r " + str(a) + " -c " + str(b) + " " + " test/a_" + str(a) + "_" + str(b) + ".d.mtx")

def genBMatrix(a, b):
    system("./gen " + " -r " + str(a) + " -c " + str(b) + " " + " test/b_" + str(a) + "_" + str(b) + ".mtx")
    system("./gen -d " + " -r " + str(a) + " -c " + str(b) + " " + " test/b_" + str(a) + "_" + str(b) + ".d.mtx")


def rmAMatrix(a, b):
    system("rm test/a_" + str(a) + "_" + str(b) + ".mtx")
    system("rm test/a_" + str(a) + "_" + str(b) + ".d.mtx")

def rmBMatrix(a, b):
    system("rm test/b_" + str(a) + "_" + str(b) + ".mtx")
    system("rm test/b_" + str(a) + "_" + str(b) + ".d.mtx")





def benchmarkTSM2():
    csvfile = open("results.csv", "w")
    compileRegularProgram()
    csvwo = csv.writer(csvfile)
    writeCSVHeader(csvwo)
    n = [10240, 15360, 20480, 25600, 30720]
    k = [2, 4, 6, 8, 16]
    for a in n:
        for b in [a//8, a//4, a//2, a]:
            for c in k:
                # Generates these matrices
                genAMatrix(a, b)
                genBMatrix(b, c)

                # Single Precision
                #t2, t3 = optParamsSingle.get((i, j))
                #compileProgram(t2, t3)
                cVal, val = testSingleValues(a, b, c)
                csvwo.writerow(["Single", a, b, c, cVal, val, val/cVal])
                # Double Precision
                #t2, t3 = optParamsDouble.get((i, j))
                #compileProgram(t2, t3)
                cVal, val = testDoubleValues(a, b, c)
                csvwo.writerow(["Double", a, b, c, cVal, val, val/cVal])

                # Removes these matrices
                rmAMatrix(a, b)
                rmBMatrix(b, c)

    csvfile.close()
            


benchmarkTSM2()
