import numpy as np
import scipy.io as sio
from utils_misc import rotation_matrix_from_vectors
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import math
import os
import sys


def getAverageGazePositionWithinFrame(timeNS, gx, gy, gz):

    #startTime = timeNS[0]

    timeSeconds = np.ceil(np.linspace(0,len(timeNS)-1, num=len(timeNS)))#(timeNS - timeNS[0])/1000000000
    #print (timeSeconds)

    totalTimeSeconds = timeSeconds[len(timeSeconds)-1]
    print (totalTimeSeconds)
    #frameNumber = np.floor(timeSeconds/30) # 30 is framerate of scene video

    #print (frameNumber[:])

    timeInGazeFrames = np.ceil(np.linspace(0,len(timeSeconds)-1, num=len(gx)))

    gxAvg = np.zeros(int(len(timeSeconds)))
    gyAvg = np.zeros(int(len(timeSeconds)))
    gzAvg = np.zeros(int(len(timeSeconds)))


    for i in range(0, len(timeSeconds)-1):
        searchval = np.ceil(timeSeconds[i])
        ii = np.where(timeInGazeFrames == searchval)[0]

        startInd = ii[0]
        endIn = ii[len(ii)-1]

        gxAvg[i] = np.mean(gx[startInd:endIn])
        gyAvg[i] = np.mean(gy[startInd:endIn])
        gzAvg[i] = np.mean(gz[startInd:endIn])

        #print (i, gxAvg[i], gyAvg[i])

    return [gxAvg, gyAvg, gzAvg]

def getFrameNumber(gx):

    frameList = np.zeros(int(len(gx)))
    frameNum = 0
    for i in range(1, len(gx)-1):

        remainder = i%200

        if remainder == 0:
            frameNum = frameNum + 1

        frameList[i] = frameNum

        print (frameNum)



    return frameList

def createList(r1, r2):
    return [item for item in range(r1, r2+1)]

# read in params
#from config import maxEcc,FL,videoRes,vidPath,gazePath,outPath, timePath

# conv to radians
#maxEcc = np.deg2rad(maxEcc)

# load por

# x=sio.loadmat(gazePath)

gazePath = sys.argv[1]
timePath = sys.argv[2]


x=pd.read_csv(gazePath)

gx = x['gaze_x_px']#.ravel()
gy = x['gaze_y_px']#.ravel()
gz = x['gaze_z_px']

t=pd.read_csv(timePath)
timeNS = t['timestamp [ns]']

[gx, gy, gz] = getAverageGazePositionWithinFrame(timeNS, gx, gy, gz)

#frameList = getFrameNumber(gx)
frameList = createList(1, len(gx))

arr = np.array([frameList, timeNS, gx, gy, gz])
arr = np.transpose(arr)

df = pd.DataFrame(arr)
df.columns = ["Frame", 'timeStamp', "gaze_x_px", "gaze_y_px", "gaze_z_px"]

df.to_csv("averageGazeWithinFrame.csv")
