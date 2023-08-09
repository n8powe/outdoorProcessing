#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:31:06 2019

@author: karl
"""


import numpy as np
import scipy.io as sio
from utils_misc import rotation_matrix_from_vectors
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import math



def processData(idx,gx,gy,FL,straightGazeVec,baseVecs,blankFrame,videoRes,outPath,vidPath,vid, frameList, gazeMag, prevGazeVec):
    
    print('Frame '+str(idx)+' of '+str(len(gx)))
    
    # this gaze vec
    

    gazeMag = np.sqrt(pow(gx[idx]-gx[idx-1], 2)+pow(gy[idx]-gy[idx-1], 2))

    #print (gazeMag)
    if gazeMag > 100:
        gazeVec = np.array([gx[idx],gy[idx],FL])
    else:
        gazeVec = prevGazeVec + (prevGazeVec - np.array([gx[idx-1],gy[idx-1],FL]))
        gazeVec[2] = FL

    #print (gazeVec)
    gazeVec = gazeVec/np.linalg.norm(gazeVec)

    # this rotm
    rotm = rotation_matrix_from_vectors(straightGazeVec,gazeVec)
    
    # rotate probe vecs
    rotatedEyeVecs = np.transpose(np.matmul(rotm,np.transpose(baseVecs)))

    # determine x and y coords of eye locations
    d = FL/rotatedEyeVecs[:,2]
    xCoords = (np.round(np.multiply(d,rotatedEyeVecs[:,0])).astype(int)+(1600/2)).astype(int)
    yCoords = (np.round(np.multiply(d,rotatedEyeVecs[:,1])).astype(int)+(1200/2)).astype(int)
    
    # create videoreader obj and writer
    
    cap = cv2.VideoCapture(vidPath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
   # # go to correct frame and read frame   
    #cap.set(1,frameList[idx])
    #print (frameList[idx])
    cap.set(1, idx)
    _,frame = cap.read()

    
    
    # read frame
   # frame = cv2.imread(vidPath+str(idx+1)+'.png')
    
    # init blank frame
    thisRetFrame = blankFrame.copy()
    
    # calc indeces
    oobIdx = np.logical_or(np.clip(xCoords,0,1599)!=xCoords,np.clip(yCoords,0,1199)!=yCoords)
    #print(len(xCoords))
    xCoords = np.clip(xCoords,0,1599)
    yCoords = np.clip(yCoords,0,1199)

    #print(len(xCoords))
    #print(np.array([yCoords,xCoords]).shape, frame.shape[:-1])

    lindex = np.ravel_multi_index(np.array([yCoords,xCoords]),frame.shape[:-1])
    silenceThese = oobIdx.reshape((videoRes,videoRes))

    for color in range(3):
        thisColor = frame[:,:,color]
        thisColor = thisColor.ravel()
        inColor = thisColor[lindex]
        inColor = np.reshape(inColor,[videoRes,videoRes])
        inColor[silenceThese] = 0
        thisRetFrame[:,:,color] = inColor
    
    thisRetFrame = np.uint8(thisRetFrame)


    vid.write(thisRetFrame)

    return [gazeMag, gazeVec]

def getAverageGazePositionWithinFrame(timeNS, gx, gy):
    
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
    

    
    for i in range(0, len(timeSeconds)-1):
        searchval = np.ceil(timeSeconds[i])
        ii = np.where(timeInGazeFrames == searchval)[0]

        startInd = ii[0]
        endIn = ii[len(ii)-1]

        gxAvg[i] = np.mean(gx[startInd:endIn])
        gyAvg[i] = np.mean(gy[startInd:endIn])

        #print (i, gxAvg[i], gyAvg[i])
         
    return [gxAvg, gyAvg]

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

# read in params
from config import maxEcc,FL,videoRes,vidPath,gazePath,outPath, timePath

# conv to radians
maxEcc = np.deg2rad(maxEcc)

# load por

# x=sio.loadmat(gazePath)

x=pd.read_csv(gazePath)

gx = x['gaze_x_px']#.ravel()
gy = x['gaze_y_px']#.ravel()

t=pd.read_csv(timePath)
timeNS = t['timestamp [ns]']

[gx, gy] = getAverageGazePositionWithinFrame(timeNS, gx, gy)

frameList = getFrameNumber(gx)

# center gaze
gx = gx-1600/2
gy = gy-1200/2
#%% create retina grid vecs

# pixel grid
xx,yy = np.meshgrid(np.arange(1,videoRes+1),np.arange(1,videoRes+1))
xx = xx-np.ceil(videoRes/2)
yy = yy-np.ceil(videoRes/2)

# convert to visual angle coords
theta = np.arctan2(yy,xx)
rho = np.multiply(np.divide(np.sqrt(np.power(xx,2)+np.power(yy,2)),np.round(videoRes/2)),maxEcc)

# convert to 3d vectors
vx = np.cos(theta)*np.sin(rho)
vy = np.sin(theta)*np.sin(rho)
vz = np.cos(rho)

baseVecs = np.concatenate((np.expand_dims(vx.ravel(),1),np.expand_dims(vy.ravel(),1),np.expand_dims(vz.ravel(),1)),axis=1)
#%% iterate over world frames and calculate mappings

# for calcuating rotations
straightGazeVec = [0,0,1]

# blank ret frame
blankFrame = np.zeros((videoRes,videoRes,3)).astype(int)
oneColor = np.zeros((videoRes,videoRes))



fourcc = cv2.VideoWriter_fourcc(*'mp4v')

vid = cv2.VideoWriter(outPath,fourcc,30,(videoRes,videoRes))


prevGazeMag = 0
prevGazeVec = 0
#for idx in range(len(gx)):
for idx in range(15000,15200):
    [prevGazeMag, prevGazeVec] = processData(idx,gx,gy,FL,straightGazeVec,baseVecs,blankFrame,videoRes,outPath,vidPath,vid, frameList, prevGazeMag, prevGazeVec)

vid.release()