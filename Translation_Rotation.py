#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:02:28 2019

@author: amithalawrence
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ipdb
############## Problem 4 - 2D Transformations on Images ################
############## Problem 4.1 - Quadrilateral transformation ##############

########### Draw the quadrilateral ############
########### Define the image space ############
img = np.zeros((300,300), np.uint8)

########### Draw the quadrilateral ############
a, b, c, d = (130,110),(100,175),(150,200),(190,175)
points = np.array([a, b, c, d], np.int32)
img_f = cv2.fillPoly(img, [points], 255)

cv2.imwrite('/Users/amithalawrence/Documents/Matlab/Quadrilateral.jpg',img_f)

########### Display the quadrilateral ##########
cv2.namedWindow('Quadrilateral Image',cv2.WINDOW_AUTOSIZE)
cv2.imshow('Image',img_f)
cv2.waitKey(0)

dx = 30
dy = 100
M = np.int32([[1,0,dx],[0,1,dy]])

img = np.zeros((300,300), np.uint8)

points_tr = np.array((np.dot(np.c_[points, np.ones(points.shape[0])], M.T)), np.int32)
img_tr = cv2.fillPoly(img, [points_tr], 255)

cv2.imwrite('/Users/amithalawrence/Documents/Matlab/Tansformed_Quad_1.png', img_tr)

cv2.namedWindow('Translated Image',cv2.WINDOW_AUTOSIZE)
cv2.imshow('Image',img_tr)
cv2.waitKey(0)

rotate_around = (150,150)
angle = 45

matrix = cv2.getRotationMatrix2D(rotate_around, angle, 1.0)

img = np.zeros((300,300), np.uint8)

points_ro = np.array((np.dot(np.c_[points_tr, np.ones(points_tr.shape[0])], matrix.T)), np.int32)
# print(points_ro)
img_ro = cv2.fillPoly(img, [points_ro], 255)

cv2.imwrite('/Users/amithalawrence/Documents/Matlab/Tansformed_Quad.png', img_ro)

cv2.namedWindow('Rotated Image',cv2.WINDOW_AUTOSIZE)
cv2.imshow('Image',img_ro)
cv2.waitKey(0)

cv2.destroyAllWindows()
