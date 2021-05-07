#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 02:13:21 2019

@author: amithalawrence
"""

import cv2
import numpy as np
import math
import operator

def findCorners(img, window_size, k, thresh):
   
    # Find x and y derivatives
    #    dy, dx = np.gradient(img)
    # np.gradient is not giving accurate results
    dx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)
    dy = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)
    Ixx = dx**2
    Ixy = dx*dy
    Iyy = dy**2
    
    # Image dimensions
    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    # color_img = newImg
    offset = int(window_size/2)
    
    #Loop through image and find our corners
    print ("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
           
            #Calculate sum of squares
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            
            # Find the weighted sum of the window of gradient
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            #If corner response is over threshold, color the point and add to corner list
            if r > thresh:
                # Append the feature points
                cornerList.append([x, y])
                
                # Plot and highlight the corners on the image
                color_img.itemset((y, x, 0), 0)
                color_img.itemset((y, x, 1), 0)
                color_img.itemset((y, x, 2), 255)
    
    return color_img, cornerList

def discriptors(gx,gy,keypoint):
    ######### Magnitude and Phase of the gradients ###########
    mag = np.zeros(gx.shape)
    angle = np.zeros(gx.shape)
    angle = np.degrees(np.arctan2(gy,gx))%360
    mag = ((gx*gx)+(gy*gy))**.5
    
    ################## The bins for the spatial histograms ##############
    binarray = []

    for ikp in keypoint:
        try:
            x,y=ikp.pt
            x=int(x)
            y=int(y)
 # 16 * 16 window of angle
            win =angle[y-8:y+8, x-8:x+8]
 # 16 * 16 window of magnitude
            win2 =mag[y-8:y+8, x-8:x+8]
            if len(win)<16:
                continue
            x1=-2
            y1=-2
            bindup=[]
 #Getting 16 4*4 windows
            for x in range(0,4):
                x1=-2
                y1=y1+4
                for y in range(0,4):
                    x1=x1+4
                    rsmall=win[y1-2:y1+2, x1-2:x1+2] #4*4 window of magnitude and angle
                    rsmall2=win2[y1-2:y1+2, x1-2:x1+2]
                    bi=dict()
                    for k in range(0,8):
                        bi[k]=0
                    for i in range(0,4):
                        for j in range(0,4):
 # getting the magnitude for each pixel and according to the angle distribute portions of magnitude between multiple bins                           
                            no=np.uint8(rsmall[i][j]/45)
                            bi[no]=(rsmall2[i][j]/45)*((no*45+45)-rsmall[i][j])
                            if no+1 in bi:
                                bi[no+1]=(rsmall2[i][j]/45)*(rsmall[i][j]-no*45)
                            else:
                                bi[0]=(rsmall2[i][j]/45)*(rsmall[i][j]-no*45)
 
 #128 dimensional 16*8 bin
                                
                    bindup.extend(list(bi.values()))


 #Normalising the Spatial Histogram
                    
            barr=np.array(bindup,np.float32)
            div=((barr**2).sum())**.5
 #clipping the value to .2
            binarray.append(np.clip((barr/div),0,.1))
        except:
             continue
        return np.array(binarray,np.float32)

## Function For Matching Descriptors : 
    
def match(binarray,binarray2):
    templis=[0.0,0,0]
    matchess=[]
    matche=[]
    for i in range(len(binarray)):
        temp=100

        for j in range(len(binarray2)):
            k=binarray[i]-binarray2[j]
            k=(k*k).sum()
            if k<temp:
                    temp=k
                    templis=[k,i,j]                
        t=templis.copy()
        matche.append(t)

    sort = sorted(matche, key=lambda tup: tup[2])
    last_used=sort[0][2]
    xyz=[]
    mainmatche=[]

#Custom function to remove bad matches
    for num in sort:
        if num[2]==last_used:
            xyz.append(num)
        elif num[2]>last_used:
            ans=sorted(xyz, key=lambda tup: tup[0])
            mainmatche.append(ans[0])
            xyz=[]
            last_used=num[2]
            xyz.append(num)

    for t in mainmatche:
        matchess.append(cv2.DMatch(t[1],t[2],t[0]))
    return matchess

########### Main program ###########

nimg1="Quadrilateral.jpg"
nimg2="Tansformed_Quad.jpg"
I=cv2.imread("/Users/amithalawrence/Documents/Matlab/"+nimg1,0)
I2=cv2.imread("/Users/amithalawrence/Documents/Matlab/"+nimg2,0)
I_origin=cv2.imread("/Users/amithalawrence/Documents/Matlab/"+    nimg1)
I2_origin=cv2.imread("/Users/amithalawrence/Documents/Matlab/"  +  nimg2)

window_size = 3
    
# Harris constant
k = 0.04798

#Threshold value    
threshold = 12099750900

    # Find gradient
dy, dx = np.gradient(I)
dy1, dx1 = np.gradient(I2)

    # Find the corners for the original figure and transformed figure
img2, keypoint = findCorners(I,window_size,k,threshold)
img3, keypoints = findCorners(I2,window_size,k,threshold)

    # Save the image
cv2.imwrite('/Users/amithalawrence/Documents/Matlab/Corner_detected_1.png',img2) 
    # Save the image
cv2.imwrite('/Users/amithalawrence/Documents/Matlab/Corner_detected_2.png',img3)
    # 

######### Run the feature descriptor ##############   
#
binarray1=discriptors(dx,dy,keypoint)
######### The transformed image #############
binarray3=discriptors(dx1,dy1,keypoints)

######## Match the common feature ############

matches=match(binarray1,binarray3)


imgx2 = cv2.drawMatches(I_origin,keypoint,I2_origin,keypoints,matches,I2,flags=2)
 
####### Save the resultant image ############
cv2.imshow("Image1",img2)
cv2.waitKey(0)
cv2.imshow("Image2",img3)
cv2.waitKey(0)
cv2.imshow("Image4",imgx2)
cv2.waitKey(0)
cv2.imwrite("/Users/amithalawrence/Documents/Matlab/Transformation_inliers.jpg",imgx2)
cv2.destroyAllWindows()

# Write top matched corners to file

# Initialize lists
list_kp1 = []
list_kp2 = []

# For each match...
for mat in matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1,y1) = keypoint[img1_idx].pt
    (x2,y2) = keypoints[img2_idx].pt

    # Append to each list
    list_kp1.append((x1, y1))
    list_kp2.append((x2, y2))


