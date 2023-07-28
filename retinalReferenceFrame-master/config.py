#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:43:15 2019

@author: karl
"""

# videoRes x videoRes as output retinal ref video
videoRes = 1001

# furthest eccentricity in degrees (corresponds to pixels videoRes/2 away from center)
maxEcc = 45

# focal length of camera in pixels
FL = 2100/2.3232

# path to world cam frames
vidPath = "C:/Users/Mary Hayhoe/Downloads/NP-20230728T170414Z-001/NP/raw-data-export/2023_07_25/rawVid.mp4"

# path to gazeCSV (worldFrame, gazeX, gazeY columns)
gazePath = "C:/Users/Mary Hayhoe/Downloads/NP-20230728T170414Z-001/NP/raw-data-export/2023_07_25/gaze.csv"

# output path for ret pngs
outPath = "E:/outdoorOutputVid/NP_ret.mp4"
 
