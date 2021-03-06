import numpy as np
import os

def parseInputs(fname):
    pwd = os.path.dirname(os.path.realpath(__file__))
    filepath = pwd + '/data/' + fname
    # filename = os.getcwd() + '//data//' + fname
    f = open (filepath, 'r')
    line = f.readline()
    numArray = []
    charArray =[]
    while line:
        tempNumList = []
        result = [x.strip() for x in line.split(',')]
        for str in result:
            if(str.isalpha() == True):
                charArray.append(str)
            if (str.isdigit() == True):
                tempNumList.append(int(str))
        line = f.readline()
        numArray.append(tempNumList)
    f.close()
    # https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array?rq=1
    numpyNumArray = np.array([np.asarray(xi) for xi in numArray])
    numpyCharArray = np.array(charArray)
    # print(numpyNumArray)
    # print(numpyCharArray)

    return numpyNumArray, numpyCharArray