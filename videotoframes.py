# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:42:16 2020

@author: amkulk
"""

import cv2
import numpy 
import os

def MakePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Video_Path='test/pixel3xl/calibration/movie.mp4'
def Video2ImagesInterval(Video_Path,Interval):
    MakePath('/'.join(Video_Path.split('/')[0:-1])+"/RawImages")#create output path
    video=cv2.VideoCapture(Video_Path)
    def getFrame(sec):
        video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = video.read()
        # breakpoint()
        if hasFrames:
            cv2.imwrite('/'.join(Video_Path.split('/')[0:-1])+"/RawImages/image"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = Interval #//it will capture image in each 0.5 second
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)