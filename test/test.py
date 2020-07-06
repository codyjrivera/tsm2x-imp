# TSM2 and ISM2 testbed
# by Cody Rivera, 2019-2020

import subprocess
import csv
import sys
from os import system, path

# Matrix generation and removal
def genAMatrix(a, b):
    matrixName = "a_" + str(a) + "_" + str(b) + ".mtx"
    matrixDName = "a_" + str(a) + "_" + str(b) + ".d.mtx"
    if not path.exists(matrixName):
        system("../gen " + " -r " + str(a) + " -c " + str(b) + " " + matrixName)
    if not path.exists(matrixDName):
        system("../gen -d " + " -r " + str(a) + " -c " + str(b) + " " + matrixDName)

def genBMatrix(a, b):
    matrixName = "b_" + str(a) + "_" + str(b) + ".mtx"
    matrixDName = "b_" + str(a) + "_" + str(b) + ".d.mtx"
    if not path.exists(matrixName):
        system("../gen " + " -r " + str(a) + " -c " + str(b) + " " + matrixName)
    if not path.exists(matrixDName):
        system("../gen -d " + " -r " + str(a) + " -c " + str(b) + " " + matrixDName)

# Test runner and result parser
def testValues(t, alg, m, n, k):
    # Build command
    command = ["../multiply"]
    if (alg == "ISM2"):
        command.append("-i")
    
    if (t == "Double"):
        command.append("-d")
        command.append("a_" + str(m) + "_" + str(k) + ".d.mtx")
        command.append("b_" + str(k) + "_" + str(n) + ".d.mtx")
    else:
        command.append("a_" + str(m) + "_" + str(k) + ".mtx")
        command.append("b_" + str(k) + "_" + str(n) + ".mtx")
    
    command.append("out.mtx")

    # Run command
    try:
        output = subprocess.check_output(command)
        outputCVal, outputVal = 0, 0
        try:
            # Horrendously hacky
            outputCVal = float(output.decode('utf-8').splitlines()[2].split(": ")[1].split(" G")[0])
            outputVal = float(output.decode('utf-8').splitlines()[3].split(": ")[1].split(" G")[0])
        except:
            outputCVal, outputVal = 0, 0
        return outputCVal, outputVal
    except:
        print("Test failed: m=" + str(m) + " n=" + str(n) + " k=" + str(k))
        sys.exit()


# Benchmark runners
def benchmarkTSM2():
    csvfile = open("results_tsm2.csv", "w")
    csvwo = csv.writer(csvfile)
    csvwo.writerow(["Precision", "m", "n", "k", "CUBLAS GFLOPS", "TSM2 GFLOPS", "Speedup"])
    M = [10240, 15360, 20480, 25600, 30720]
    N = [2, 4, 8, 16]
    for m in M:
        # in TSM2, k ~= m
        for k in [m // 8, m // 4, m // 2, m]:
            for n in N:
                # Generates these matrices
                genAMatrix(m, k)
                genBMatrix(k, n)
                # Single Precision
                cVal, val = testValues("Single", "TSM2", m, n, k)
                csvwo.writerow(["Single", m, n, k, cVal, val, val/cVal])
                # Double Precision
                cVal, val = testValues("Double", "TSM2", m, n, k)
                csvwo.writerow(["Double", m, n, k, cVal, val, val/cVal])
    csvfile.close()

def benchmarkISM2():
    csvfile = open("results_ism2.csv", "w")
    csvwo = csv.writer(csvfile)
    csvwo.writerow(["Precision", "m", "n", "k", "CUBLAS GFLOPS", "ISM2 GFLOPS", "Speedup"])
    M = [10000, 100000, 1000000, 10000000]
    K = [2, 4, 8, 16]
    N = K
    for m in M:
        for k in K:
            for n in N:
                # Generates these matrices
                genAMatrix(m, k)
                genBMatrix(k, n)
                # Single Precision
                cVal, val = testValues("Single", "ISM2", m, n, k)
                csvwo.writerow(["Single", m, n, k, cVal, val, val/cVal])
                # Double Precision
                cVal, val = testValues("Double", "ISM2", m, n, k)
                csvwo.writerow(["Double", m, n, k, cVal, val, val/cVal])
    csvfile.close()

if (path.exists("../multiply")) and (path.exists("../gen")):
    benchmarkTSM2()
    benchmarkISM2()
else:
    print("Ensure both ../multiply and ../gen exist")