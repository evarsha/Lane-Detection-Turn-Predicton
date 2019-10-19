#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt
from scipy import ndimage, misc

c=cv2.VideoCapture('challenge_video.mp4')
K=np.array([[  1154.227,   0 ,  671.628]
, [  0,   1148.182 ,  386.0463],
 [  0 ,  0,   1]])

D = np.array([[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03 , -8.79107779e-05
,    2.20573263e-02]])


# In[2]:


def find_homography(img1, img2):
    ind = 0
    A = np.empty((8, 9))
    for pixel in range(0, len(img1)):
        x1 = img1[pixel][0]
        y1 = img1[pixel][1]
        x2= img2[pixel][0]
        y2 = img2[pixel][1]
        A[ind] = np.array([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A[ind + 1] = np.array([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
        ind = ind + 2
    U, s, V = np.linalg.svd(A, full_matrices=True)
    V = (copy.deepcopy(V)) / (copy.deepcopy(V[8][8]))
    H = V[8,:].reshape(3, 3)
    return H


# In[3]:


# sliding window concept 
def extract_plot_lane_line(histogram, input_image ,plot=True):
        new_img = np.dstack((input_image, input_image, input_image))*255
        midpoint = np.int(histogram.shape[0]/2)
        bottom_left = np.argmax(histogram[:midpoint])
        bottom_right = np.argmax(histogram[midpoint:]) + midpoint
        number_windows = 9
        set_height_window = np.int(input_image.shape[0]/number_windows)
        line_present = input_image.nonzero()
        line_presenty = np.array(line_present[0])
        line_presentx = np.array(line_present[1])
        frame_leftx = bottom_left
        frame_rightx = bottom_right
        margin = 50
        minpix = 10
        left_lane_pixels = []
        right_lane_pixels = []
        
        
        #Making Boxes for creating Sliding Window
        for window in range(number_windows):
            boundry_y_low = input_image.shape[0] - (window+1)*set_height_window
            boundry_y_high = input_image.shape[0] - window*set_height_window
            boundry_xleft_low = frame_leftx - margin
            boundry_xleft_high = frame_leftx + margin
            boundry_xright_low = frame_rightx - margin
            boundry_xright_high = frame_rightx + margin
            cv2.rectangle(new_img,(boundry_xleft_low,boundry_y_low),(boundry_xleft_high,boundry_y_high),(0,255,0), 2) 
            cv2.rectangle(new_img,(boundry_xright_low,boundry_y_low),(boundry_xright_high,boundry_y_high),(0,255,0), 2) 
            true_left = ((line_presenty >= boundry_y_low) & (line_presenty < boundry_y_high) & (line_presentx >= boundry_xleft_low) & (line_presentx < boundry_xleft_high)).nonzero()[0]
            true_right = ((line_presenty >= boundry_y_low) & (line_presenty < boundry_y_high) & (line_presentx >= boundry_xright_low) & (line_presentx <boundry_xright_high)).nonzero()[0]
            left_lane_pixels.append(true_left)
            right_lane_pixels.append(true_right)
            if len(true_left) > minpix:
                frame_leftx = np.int(np.mean(line_presentx[true_left]))
            if len(true_right) > minpix:        
                frame_rightx = np.int(np.mean(line_presentx[true_right]))
                
        #Concatenating the left and right lane pixels        
        left_lane_pixels = np.concatenate(left_lane_pixels)
        right_lane_pixels = np.concatenate(right_lane_pixels)
        left_x = line_presentx[left_lane_pixels]
        left_y = line_presenty[left_lane_pixels] 
        right_x = line_presentx[right_lane_pixels]
        right_y = line_presenty[right_lane_pixels]
        return left_x, left_y, right_x, right_y, left_lane_pixels, right_lane_pixels

#plotting the fitted lines
def poly_fit(left_x, left_y, right_x, right_y, left_lane_pixels, right_lane_pixels, sobel, plot:True):  
        left_line = np.polyfit(left_y, left_x, 2)
        right_line = np.polyfit(right_y, right_x, 2)
        ploty = np.linspace(0, sobel.shape[0]-1, sobel.shape[0] )
        left_linex = left_line[0]*ploty**2 + left_line[1]*ploty + left_line[2]
        right_linex = right_line[0]*ploty**2 + right_line[1]*ploty + right_line[2]
        nonzero = sobel.nonzero()
        line_presenty = np.array(nonzero[0])
        line_presentx = np.array(nonzero[1])
        new_img = np.dstack((sobel,sobel, sobel))*255
        new_img[line_presenty[left_lane_pixels], line_presentx[left_lane_pixels]] = [255, 0, 0]
        new_img[line_presenty[right_lane_pixels], line_presentx[right_lane_pixels]] = [0, 0, 255]
        if(plot):
            cv2.imshow("new_image",new_img)

        return new_img,left_line, right_line, ploty, left_linex, right_linex
       


# In[4]:


# calculating the curvature using two methods 
def compute_curvature(left_line, right_fit, ploty, left_linex, right_linex, left_x, left_y, rightx, righty ,sobel):
        ym_pix = 30/720 # meters per pixel in y dimension
        xm_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(ploty)
        left_cr = np.polyfit(ploty * ym_pix, left_linex * xm_pix, 2)
        curverad_left = ((1 + (2 * left_line[0] * y_eval / 2. +left_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_cr[0])
        right_cr = np.polyfit(ploty * ym_pix, right_linex * xm_pix, 2)
        curverad_right = ((1 + (2 * left_line[0] * y_eval / 2. + right_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_cr[0])
        middle_car = sobel.shape[1] / 2
        middle_lane = (left_linex[0] + right_linex[0]) / 2
        turn = (middle_lane-middle_car)*xm_pix # predicting turn using the differenceo of middle lane and car 
        mean_curvature = curverad_right+ curverad_left
        if turn<=-0.1:
            print_turn ="Turning Left"
        elif turn>=0.1:
            print_turn ="Turning Right"
        else:
            print_turn ="No Turn"
        return print_turn 


# In[5]:


def show_on_real_world(turn,image, sobel, Minv, left_linex, right_linex, ploty): # function plot the values in image frame 
    font= cv2.FONT_HERSHEY_SIMPLEX
    color = np.zeros_like(sobel).astype(np.uint8)
    points_left = np.array([np.transpose(np.vstack([left_linex, ploty]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_linex, ploty])))])
    points = np.hstack((points_left, points_right))
    cv2.fillPoly(color, np.int_([points]), (0,255,0)) # filling the lane region 
    newwarp = cv2.warpPerspective(color, Minv, (image.shape[1], image.shape[0]))
    cv2.putText(newwarp,str(turn),(630, 500), font, 0.7,(255,0,255),2,cv2.LINE_AA) # printhing the predicted turn 
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0) # adding both the images 
    return result


# In[6]:


while (True):
    ret,image=c.read()
    if ret == True:   
        undistort=cv2.undistort(image,K,D,None,K)  #undistorting the image 
        Hpoints=np.array([[560,450], [740, 450], [95,710],[1260, 710]])
        H = find_homography(Hpoints, [[0,0],[254,0],[0,254],[254,254]]) # finding the homography matrix 
        Hinv=np.linalg.inv(H) # inverse of homography 
        im_out = cv2.warpPerspective(undistort, H, (255,255)) # getting the warped image
        red_channel = im_out[:,:,2] #extracting the red channel 
        ret, thresh = cv2.threshold(red_channel, 170, 255, cv2.THRESH_BINARY) #binary threshing 
        sobelx = cv2.Sobel(thresh, cv2.CV_64F, 1, 0) # sobel in x
        sobely = cv2.Sobel(thresh, cv2.CV_64F, 0, 1) # sobel in y 
        sobel = np.sqrt((sobelx**2) + (sobely**2))
        histogram = np.sum(sobel[sobel.shape[0]//2:,:], axis=0)

        left_x, left_y, right_x, right_y, left_lane_pixels, right_lane_pixels=extract_plot_lane_line(histogram, sobel)
        if left_x!=[] and right_x!=[]:
            new_img,left_line, right_line, ploty, left_linex, right_linex=poly_fit(left_x, left_y, right_x, right_y, left_lane_pixels, right_lane_pixels, sobel,True)
            value=compute_curvature(left_line, right_line, ploty, left_linex, right_linex, left_x, left_y, right_x, right_y,sobel)
            original=show_on_real_world(value,image, new_img, Hinv, left_linex, right_linex, ploty)
            cv2.imshow("Original",original)
        k = cv2.waitKey(10)
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
    #i=i+1
c.release
cv2.destroyAllWindows()
