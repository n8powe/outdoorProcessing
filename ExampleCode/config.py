#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:43:15 2019

@author: karl
"""

# videoRes x videoRes as output retinal ref video
videoRes = 1201

# furthest eccentricity in degrees (corresponds to pixels videoRes/2 away from center)
maxEcc = 45

# focal length of camera in pixels
FL = 2100/2.2

# path to world cam frames
vidPath = "C:/Users/natha/OneDrive/Desktop/2023_07_25/NP/raw-data-export/2023-07-25/rawVid.mp4"

# path to gazeCSV (worldFrame, gazeX, gazeY columns)
gazePath = "C:/Users/natha/OneDrive/Desktop/2023_07_25/NP/raw-data-export/2023-07-25/gaze.csv"

# output path for ret pngs
outPath = "C:/Users/natha/OneDrive/Desktop/2023_07_25/NP/raw-data-export/2023-07-25/NP_ret.mp4"

# World time 

timePath = "C:/Users/natha/OneDrive/Desktop/2023_07_25/NP/raw-data-export/2023-07-25/world_timestamps.csv"
 
