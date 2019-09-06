# Brute force parameter optimization for TSM2
# by Cody Rivera

import subprocess
import csv

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
    wo.writerow(["Precision", "n", "k", "t1", "t2", "t3", "CUBLAS GFLOPS", "TSM2 GFLOPS", "Speedup"])

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


# Compiles the program
def compileProgram(t2, t3):
    writeParameterFile(t1Value, t2, t3)
    subprocess.check_output(["make"], stderr=subprocess.DEVNULL)


# Testbench function

def testSingleValues(n, k, t2, t3):
    try:
        single = subprocess.check_output(["ssh", "gpu01", "cd re*/ts* ; ./multiply test/a_" + str(n) + "_" + str(n) + ".mtx "
                             + "test/b_" + str(n) + "_" + str(k) + ".mtx out"])
        
        singleCVal, singleVal = 0, 0
        try:
            # Horrendously hacky
            singleCVal = float(single.decode('utf-8').splitlines()[2].split(": ")[1].split(" G")[0])
            singleVal = float(single.decode('utf-8').splitlines()[3].split(": ")[1].split(" G")[0])
        except:
            singleCVal, singleVal = 0, 0
    
        return singleCVal, singleVal
    except:
        compileProgram(t2, t3)
        return testSingleValues(n, k, t2, t3)


def testDoubleValues(n, k, t2, t3):
    try:
        double = subprocess.check_output(["ssh", "gpu01", "cd re*/ts* ; ./multiply -d test/a_" + str(n) + "_" + str(n) + ".d.mtx "
                             + "test/b_" + str(n) + "_" + str(k) + ".d.mtx out"])

        doubleCVal, doubleVal = 0, 0
        
        try:
            # Horrendously hacky
            doubleCVal = float(double.decode('utf-8').splitlines()[2].split(": ")[1].split(" G")[0])
            doubleVal = float(double.decode('utf-8').splitlines()[3].split(": ")[1].split(" G")[0])
        except:
            doubleCVal, doubleVal = 0, 0
    
        return doubleCVal, doubleVal
    except:
        compileProgram(t2, t3)
        return testDoubleValues(n, k, t2, t3)



def benchmarkTSM2():
    csvfile = open("results.csv", "w")
    csvwo = csv.writer(csvfile)
    writeCSVHeader(csvwo)
    n = [10240, 15360, 20480, 25600, 30720]
    k = [2, 4, 6, 8, 16]
    for i in n:
        for j in k:
            # Single Precision
            t2, t3 = optParamsSingle.get((i, j))
            compileProgram(t2, t3)
            cVal, val = testSingleValues(i, j, t2, t3)
            csvwo.writerow(["Single", i, j, t1Value, t2, t3, cVal, val, val/cVal])
            # Double Precision
            t2, t3 = optParamsDouble.get((i, j))
            compileProgram(t2, t3)
            cVal, val = testDoubleValues(i, j, t2, t3)
            csvwo.writerow(["Double", i, j, t1Value, t2, t3, cVal, val, val/cVal])
    csvfile.close()
            


benchmarkTSM2()
