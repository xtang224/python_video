#!/usr/bin/env python

'''
usage: python JudgeAndDraw.py yourTest.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from numpy import *
import cv2
import io
import math
from os import listdir
import sys, getopt

def resize(img):
    mm = 9
    nn = 9
    v = img.shape
    rows = v[0]
    cols = v[1]
    img2 = np.zeros((9,9),np.float)
    rowRatio = rows / 9
    colRatio = cols / 9
    for i in range(rows):
       for j in range(cols):        
          newRow = int(i / rowRatio)
          if newRow>8:
             newRow = 8          
          newCol = int(j / colRatio)
          if newCol > 8: 
             newCol = 8
          #newAve = (img[i,j][0]*0.3 + img[i,j][1]*0.3 + img[i,j][2]*0.3)
          newAve = img[i,j]
          img2[newRow, newCol] = img2[newRow, newCol] + newAve
    for i in range(mm):
       for j in range(nn):
          img2[i,j] = img2[i,j] / (rows/mm * cols/nn)
          #print ("i=%d,j=%d,img[i,j]=%d" % (i,j,img2[i,j]))
          #img2[i,j] -= 192
    print ("img2.size=", img2.size)
    return img2

def gray(img):
   tmpImg = resize(img)
   #img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   img3 = tmpImg
   sum = 0
   v = img3.shape
   rows = v[0]
   cols = v[1]
   for i in range(rows):
       for j in range(cols):
          sum += img3[i,j]
   avg = sum / (rows*cols)
   for i in range(rows):
       for j in range(cols):
          #if img3[i,j]>0 and img3[i,j]<3 and j>14:
          if img3[i,j] >= avg:
             img3[i,j] = 1
	  else:
	     img3[i,j] = 0
		 
   result = ""
   for i in range(rows):
       result = ""
       for j in range(cols):
	  result += str(img3[i,j])
       #print (result)
   print("img3.size =", img3.size)
   img3 = img3.reshape(1,img3.size)

   result = ""  
   for i in range(img3.size):
      result = result + str(int(img3[0,i]))      
   #print (result)
   f = open("temp", "w")
   f.write(result)
   f.close()
   return img3
   
def resize2(img):
    mm = 32
    nn = 32
    v = img.shape
    rows = v[0]
    cols = v[1]
    img2 = np.zeros((32,32),np.float)
    rowRatio = rows / 32
    colRatio = cols / 32
    for i in range(rows):
       for j in range(cols):        
          newRow = int(i / rowRatio)
          if newRow>31:
             newRow = 31          
          newCol = int(j / colRatio)
          if newCol > 31: 
             newCol = 31
          #newAve = (img[i,j][0]*0.3 + img[i,j][1]*0.3 + img[i,j][2]*0.3)
          newAve = img[i,j]
          img2[newRow, newCol] = img2[newRow, newCol] + newAve
    for i in range(mm):
       for j in range(nn):
          img2[i,j] = img2[i,j] / (rows/mm * cols/nn)
          #print ("i=%d,j=%d,img[i,j]=%d" % (i,j,img2[i,j]))
          #img2[i,j] -= 192
    print ("img2.size=", img2.size)
    return img2, rowRatio, colRatio

def gray2(img):
   tmpImg, rowTimes, colTimes = resize2(img)
   #img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   img3 = tmpImg
   sum = 0
   v = img3.shape
   rows = v[0]
   cols = v[1]
   for i in range(rows):
       for j in range(cols):
          sum += img3[i,j]
   avg = sum / (rows*cols)
   for i in range(rows):
       for j in range(cols):
          #if img3[i,j]>0 and img3[i,j]<3 and j>14:
          if img3[i,j] >= avg:
             img3[i,j] = 1
	  else:
	     img3[i,j] = 0
		 
   result = ""
   for i in range(rows):
       result = ""
       for j in range(cols):
	  result += str(img3[i,j])
       #print (result)
   print("img3.size =", img3.size)
   img3 = img3.reshape(1,img3.size)

   result = ""  
   for i in range(img3.size):
      result = result + str(int(img3[0,i]))      
   #print (result)
   f = open("temp1024", "w")
   f.write(result)
   f.close()
   return img3, rowTimes, colTimes

def getContent(fileName, tag):
    strContent = ""
    f = open(fileName)
    contents = f.readlines()
    f.close()
    for i in range(len(contents)):
        content = contents[i]
        strContent += content
    #print ("strContent = ", strContent)
    startTag = "<" + tag + ">"
    endTag = "</" + tag + ">"
    startIndex = strContent.find(startTag)
    endIndex = strContent.find(endTag)
    startIndex = startIndex + len(startTag)
    request = strContent[startIndex:endIndex]
    #print ("request = ", request)
    return request

def loadBases():    
    bases = []
    baseRatios = []
    starts = []
    fileName = "sortGraph.xml"
    for i in range(135): #there are 135 base files for drawing the rectangle  
       tag = "base" + str(i+1)	
       base = getContent(fileName, tag)       
       bases.append(base)
       tag = "basestart" + str(i+1)	
       start = getContent(fileName, tag)        
       starts.append(start)
       tag = "baseratio" + str(i+1)	
       ratio = getContent(fileName, tag)          
       baseRatios.append(ratio)
    return bases, starts, baseRatios
	
def loadTrain():
    dataArr = mat(zeros((150,81)))
    #labelArr = mat(zeros((150,1)))
    labelArr = []
    fileName = "sortGraph.xml"
    for i in range(150): #there are 150 train files for using the Support Vector Machine Algorithm
       tag = "train" + str(i+1)	
       train = getContent(fileName, tag)       
       dataArr[i,:] = train
       tag = "trainlabel" + str(i+1)	
       label = getContent(fileName, tag)        
       labelArr.append(float(label))
    return dataArr, labelArr
	
def loadTest():
    dataArr = mat(zeros((1,81)))
    #labelArr = mat(zeros((150,1)))
    labelArr = []
    fileName = "sortGraph.xml"
    for i in range(6): #there are 6 test files for using the Support Vector Machine Algorithm
       tag = "test" + str(i+1)	
       test = getContent(fileName, tag)       
       dataArr[i,:] = test
       tag = "testlabel" + str(i+1)	
       label = getContent(fileName, tag)        
       labelArr.append(float(label))
    return dataArr, labelArr
	
def loadTest1024():
    dataArr = []
    labelArr = []
    rowTimes = []
    colTimes = []
    fileName = "sortGraph.xml"
    for i in range(6): #there are 6 test files for using the Support Vector Machine Algorithm
       tag = "test1024_" + str(i+1)	
       test = getContent(fileName, tag)       
       dataArr.append(test)
       tag = "testlabel" + str(i+1)	
       label = getContent(fileName, tag)        
       labelArr.append(float(label))
       tag = "test1024ratio" + str(i+1)	
       testratio = getContent(fileName, tag) 
       rowTimes.append(float(testratio.split(",")[0]))
       colTimes.append(float(testratio.split(",")[1]))
    return dataArr, labelArr, rowTimes, colTimes
	
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m = shape(X)[0]
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K
	   
def testDigits(label, kTup=('rbf', 10)):
    fileName = "sortGraph.xml"
    dataArr, labelArr = loadTrain()
    bSave = getContent(fileName, "b")
    b = float(bSave)
    #get svInd
    svIndSave = getContent(fileName, "svInd")
    svInd_pre = svIndSave.split(",")
    svInd = []
    for i in range(len(svInd_pre)):
       svInd.append(int(svInd_pre[i]))
    #get alphas   
    alphaSave = getContent(fileName, "alphas")
    alpha_pre = alphaSave.split(",")
    alphas = mat(zeros((len(alpha_pre),1)))
    for i in range(len(alpha_pre)):
       alphas[i,:] = float(alpha_pre[i])
	
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()    
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd]
    print ("there are %d Support Vectors" % (shape(sVs)[0]))
    m = shape(datMat)[0]
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))	

    dataArr = zeros((1,81))
    f = open("temp")
    line = f.readline()
    f.close()
    dataArr[0,:] = line
    labelArr = [label]
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m = shape(datMat)[0]
    isRedLight = []
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
	print ("i=%d" % (i))
	print ("predict = ", predict)
        if sign(predict) > 0:
	   isRedLight.append(i)
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print ("the test error rate is: %f" % (float(errorCount)/m))

    return isRedLight


def getDistance(line1, line2):
    dist = 0
    #print ("line1 = ", line1)
    #print ("line2 = ", line2)
    for i in range(len(line1)):
       if int(line2[i])==1:
          if int(line1[i])!=int(line2[i]):
             dist += 1
    return dist
	
def draw_rects(img, rects, color):
    x1 = int(rects[0])
    y1 = int(rects[1])
    x2 = int(rects[2])
    y2 = int(rects[3])
    #for x1, y1, x2, y2 in rects:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    if len(sys.argv)<3:
        print ("usage: python JudgeAndDraw.py yourPic.jpg label")
	print ("where label is 1, if yourPic.jpg contains a redlight, otherwise -1")
	print ("for example, python JudgeAndDraw.py redLightTest.jpg 1")
	exit()
    args = getopt.getopt(sys.argv[1:], "")
    print (args)

    img = cv2.imread(args[1][0], 0)
    gray(img)
    img2, rowTimes, colTimes = gray2(img)
    print ("rowTimes = %d, colTimes = %d" % (rowTimes, colTimes))
	
    label_in = args[1][1]
    label = int(label_in)
	
    bases, xys, baseRatios = loadBases()
    m = shape(bases)[0]
    isRedLightIndexes = testDigits(label)
	
    if label==-1:
        print ("the picture does not contain a redlight, therefore we will not locate it with a rectangle")
	exit()

    testData = []
    f = open("temp1024")
    line = f.readline()
    f.close()
    testData.append(line)
    for i in range(len(isRedLightIndexes)):
       lowestDis = 10000
       lowestIndex = 0
       for j in range(m):
          dist = getDistance(testData[isRedLightIndexes[i]], bases[j])
          if dist < lowestDis:
             lowestDis = dist
	     lowestIndex = j
       #now we want to find possible multiple red lights	
       print ("lowestDis = %f and lowestIndex=%d" % (lowestDis, lowestIndex))
       lowestIndexes = [lowestIndex]
       for j in range(m):
          dist = getDistance(testData[isRedLightIndexes[i]], bases[j])
	  if dist==lowestDis:
	     lowestIndexes.append(j)	   
       #now we are going to highlight the red lights detected
       fileName = args[1][0]
       img = cv2.imread(fileName, 0)
       for j in range(len(lowestIndexes)):
	  lowestIndex = lowestIndexes[j]
	  xy = xys[lowestIndex]
          x1 = float(xy.split(",")[0])
          y1 = float(xy.split(",")[1])
          #x1 = lowestIndex / 7 *7 * colTimes[i]
          #y1 = lowestIndex % 7 *7 * rowTimes[i]
          rects = [x1*colTimes, y1*rowTimes, x1*colTimes+8.89*colTimes, y1*rowTimes+21.33*rowTimes]
          draw_rects(img, rects, (0, 0, 255)) 
       cv2.imshow('capture %d' % isRedLightIndexes[i], img)       
       cv2.waitKey()
    cv2.destroyAllWindows()


    
