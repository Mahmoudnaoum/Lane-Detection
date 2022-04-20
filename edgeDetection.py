import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
  
def binaryArray(array, thresh, value=0):
  # Return a 2D binary array in which all pixels are either 0 or 1
     
  if value == 0:
    binary = np.ones_like(array) 
         
  else:
    binary = np.zeros_like(array)  
    value = 1

  binary[(array >= thresh[0]) & (array <= thresh[1])] = value
 
  return binary
 
def blurGaussian(channel, ksize=3):
  # Implementation for Gaussian blur to reduce noise 

  return cv2.GaussianBlur(channel, (ksize, ksize), 0)
         
def magThresh(image, sobelKernel=3, thresh=(0, 255)):
  # Implementation of Sobel edge detection

  sobelx = np.abs(sobel(image, orient='x', sobelKernel=sobelKernel))

  sobely = np.abs(sobel(image, orient='y', sobelKernel=sobelKernel))
 
  mag = np.sqrt(sobelx ** 2 + sobely ** 2)
  
  return binaryArray(mag, thresh)
 
def sobel(imgChannel, orient='x', sobelKernel=3):
  # Find edges that are aligned vertically and horizontally on the image

  if orient == 'x':
    sobel = cv2.Sobel(imgChannel, cv2.CV_64F, 1, 0, sobelKernel)

  if orient == 'y':
    sobel = cv2.Sobel(imgChannel, cv2.CV_64F, 0, 1, sobelKernel)
 
  return sobel
 
def threshold(channel, thresh=(128,255), threshType=cv2.THRESH_BINARY):
  # Apply a threshold to the input channel

  return cv2.threshold(channel, thresh[0], thresh[1], threshType)