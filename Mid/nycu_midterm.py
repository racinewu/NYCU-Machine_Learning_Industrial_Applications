import numpy as np


def read_txt(inFileName, numOfFeature):
    # init
    recArr = []
    clsArr = []

    with open(inFileName, "r") as inFile:
        #%%% read features
        s = inFile.readline()  # skip comment line

        for ic in range(numOfFeature):
            s = inFile.readline()
            value = [
                0.
            ]  # since we use append, value must be created in the loop
            data1 = s.strip()  # remove leading and ending blanks
            if (len(data1) <= 0):
                break

            data1 = data1.replace('[', '')  # remove [
            data1 = data1.replace(']', '')  # remove ]
            strs = data1.split()  # array of 1 str

            value[0] = eval(strs[0])  # convert to real
            recArr.append(value)
            # add 1 record at ending
        # end for

        s = inFile.readline()  # skip comment line

        while True:
            s = inFile.readline()
            data1 = s.strip()  # remove leading and ending blanks
            if (len(data1) <= 0):
                break

            data1 = data1.replace('[', '')  # remove [
            data1 = data1.replace(']', '')  # remove ]
            strs = data1.split()  # array of 26 str

            for t in strs:
                clsArr.append(eval(t))
        # end while

    npXY = np.array(recArr)
    npC = np.array(clsArr)

    return npXY, npC


X, y = read_txt("wave60_dataset.txt", 60)
