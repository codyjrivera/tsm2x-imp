# Brute force parameter optimization for TSM2
# by Cody Rivera

import subprocess

diff = .1

t1Value = 128

nRounds = 5

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

def testValues(n, k, t2, t3):
    try:
        single = subprocess.check_output(["ssh", "gpu01", "cd re*/ts* ; ./multiply test/a_" + str(n) + "_" + str(n) + ".mtx "
                             + "test/b_" + str(n) + "_" + str(k) + ".mtx out"])
        double = subprocess.check_output(["ssh", "gpu01", "cd re*/ts* ; ./multiply -d test/a_" + str(n) + "_" + str(n) + ".d.mtx "
                             + "test/b_" + str(n) + "_" + str(k) + ".d.mtx out"])
        singleVal = 0
        doubleVal = 0
        try:
            # Horrendously hacky
            singleVal = float(single.decode('utf-8').splitlines()[3].split(": ")[1].split(" G")[0])
        except:
            singleVal = 0
        try:
            # Horrendously hacky
            doubleVal = float(double.decode('utf-8').splitlines()[3].split(": ")[1].split(" G")[0])
        except:
            doubleVal = 0
    
        return singleVal, doubleVal
    except:
        compileProgram(t2, t3)
        return testValues(n, k, t2, t3)



# Optimize for a single n and k

def optimizeParameters(n, k):
    sMaxT2, sMaxT3, sMax = 4, 4, 0
    dMaxT2, dMaxT3, dMax = 4, 4, 0
    sVal, dVal, sValT, dValT = 0, 0, 0, 0
    for t2 in range(k, k + 1):
        for t3 in range(4, t1Value + 1, 4):
            compileProgram(t2, t3)
            sValT, dValT = 0, 0
            for _ in range(nRounds):
                sVal, dVal = testValues(n, k, t2, t3)
                sValT, dValT = sValT + sVal, dValT + dVal
            sVal, dVal = sValT / nRounds, dValT / nRounds
            print("INTERMEDIATE t2: {}, t3: {}, single: {}, double: {}".format(t2, t3, sVal, dVal))  
            if (sVal > sMax):
                sMaxT2 = t2
                sMaxT3 = t3
                sMax = sVal
            else:
                pass
            if (dVal > dMax):
                dMaxT2 = t2
                dMaxT3 = t3
                dMax = dVal
            else:
                pass
    return (sMaxT2, sMaxT3) , (dMaxT2, dMaxT3) 


def optimizeTSM2():
    sResults, dResults = None, None
    print("       n   k    Single:  t2  t3  Double:  t2  t3")
    n = [10240, 15360, 20480, 25600, 30720]
    k = [2, 4, 6, 8, 16]
    for i in n:
        for j in k:
            sResults, dResults = optimizeParameters(i, j)
            print("{:8} {:3}            {:3} {:3}          {:3} {:3}".format(i, j, sResults[0], sResults[1], dResults[0], dResults[1])) 
            


optimizeTSM2()
