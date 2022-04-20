import cv2 # Import the OpenCV library to enable computer vision
import numpy as np
from sympy import false # Import the NumPy scientific computing library
import edgeDetection as edge # Handles the detection of lane lines
import matplotlib.pyplot as plt # Used for plotting and error checking
import sys # Used for path file in command
import re # Used to do some Regex

class imageLaneDetect:

  def __init__(self, origFrame):

    self.origFrame = origFrame
       
    self.laneLineMarkings = None
 
    self.warpedFrame = None
    self.transformationMatrix = None
    self.invTransformationMatrix = None
 
    self.origImageSize = self.origFrame.shape[::-1][1:]
 
    width = self.origImageSize[0]
    height = self.origImageSize[1]
    self.width = width
    self.height = height
     
    # Four corners of the trapezoid-shaped region of interest
    # self.roiPoints = np.float32([
    #   (679,613), # Top-left corner
    #   (329,855), # Bottom-left corner            
    #   (1306,855), # Bottom-right corner
    #   (690,450) # Top-right corner
    # ])

    self.roiPoints = np.float32([
      (570,460),
      (213,684),
      (1120,680),
      (715,460)
      ])
    # After we perform perspective transformation.
    self.padding = int(0.25 * width) 
    self.desiredRoiPoints = np.float32([
      [self.padding, 0], # Top-left corner
      [self.padding, self.origImageSize[1]], # Bottom-left corner         
      [self.origImageSize[0]-self.padding, self.origImageSize[1]], # Bottom-right corner
      [self.origImageSize[0]-self.padding, 0] # Top-right corner
    ]) 
         
    self.histogram = None

    self.noOfWindows = 10
    self.margin = int((1/12) * width)  
    self.minpix = int((1/24) * width)  
         
    # Best fit lines for left and right of the lane
    self.leftFit = None
    self.rightFit = None
    self.leftLaneInds = None
    self.rightLaneInds = None
    self.ploty = None
    self.leftFitx = None
    self.rightFitx = None
    self.leftx = None
    self.rightx = None
    self.lefty = None
    self.righty = None
         
    # Pixel parameters for x and y dimensions
    self.YMPERPIX = 4.67 / 267 # meters per pixel in y dimension
    self.XMPERPIX = 2.08 / 143 # meters per pixel in x dimension
         
    # Radius of curvature and offset
    self.leftCurvem = None
    self.rightCurvem = None
    self.centerOffset = None
 
  def calculateCarPosition(self, printToTerminal=False):
    # Calculate the position of the car relative to the center

    # Camera in the middle of the image
    carLocation = self.origFrame.shape[1] / 2
 
    height = self.origFrame.shape[0]
    bottomLeft = self.leftFit[0]*height**2 + self.leftFit[1]*height + self.leftFit[2]
    bottomRight = self.rightFit[0]*height**2 + self.rightFit[1]*height + self.rightFit[2]
    centerLane = (bottomRight - bottomLeft)/2 + bottomLeft 
    centerOffset = (np.abs(carLocation) - np.abs(centerLane)) * self.XMPERPIX
 
    if printToTerminal == True:
      offsetStr = ""
      if (centerOffset < 0):
        offsetStr = "left of center"
      elif (centerOffset > 0):
        offsetStr = "right of center"
      print(f"Center offset= {np.abs(centerOffset)}m {offsetStr}.")
             
    self.centerOffset = centerOffset
       
    return centerOffset
 
  def calculateCurvature(self, printToTerminal=False):
    # Calculate the road curvature in meters

    # Bottom of the frame.
    yEval = np.max(self.ploty)    
 
    # Fit curves to the real world
    leftFitCr = np.polyfit(self.lefty * self.YMPERPIX, self.leftx * (self.XMPERPIX), 2)
    rightFitCr = np.polyfit(self.righty * self.YMPERPIX, self.rightx * (self.XMPERPIX), 2)
             
    # Calculate the radius of curvature
    leftCurvem = ((1 + (2*leftFitCr[0]*yEval*self.YMPERPIX + leftFitCr[1])**2)**1.5) / np.absolute(2*leftFitCr[0])
    rightCurvem = ((1 + (2*rightFitCr[0]*yEval*self.YMPERPIX + rightFitCr[1])**2)**1.5) / np.absolute(2*rightFitCr[0])

    if printToTerminal == True:
      print(f"Left lane curvature= {leftCurvem}m.")
      print(f"Right lane curvature= {rightCurvem}m.")
      print(f"The radius of curvature= {(leftCurvem+rightCurvem)/2}m.")
             
    self.leftCurvem = leftCurvem
    self.rightCurvem = rightCurvem
 
    return leftCurvem, rightCurvem        
         
  def calculateHistogram(self,frame=None,plot=False):
    # Calculate the image histogram to find peaks in white pixel count

    if frame is None:
      frame = self.warpedFrame
             
    # Generate the histogram
    self.histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)
 
    if plot == True:
      figure, (ax1, ax2) = plt.subplots(2,1)
      figure.set_size_inches(10, 5)
      ax1.imshow(frame, cmap='gray')
      ax1.set_title("Warped Binary Frame")
      ax2.plot(self.histogram)
      ax2.set_title("Histogram Peaks")
      plt.show()
             
    return self.histogram
 
  def displayCurvatureOffset(self, frame=None, plot=False):
    # Display curvature and offset statistics on the image

    imageCopy = None
    if frame is None:
      imageCopy = self.origFrame.copy()
    else:
      imageCopy = frame
 
    cv2.putText(imageCopy,'Curve Radius: '+str((self.leftCurvem+self.rightCurvem)/2)[:7]+'m', (int((5/600)*self.width), int((20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*self.width)),(255,255,255),2,cv2.LINE_AA)

    offsetStr = ""
    if (self.centerOffset < 0):
      offsetStr = " Left of Center"
    elif (self.centerOffset > 0):
      offsetStr = " Right of Center"

    cv2.putText(imageCopy,'Center Offset: '+str(np.abs(self.centerOffset))[:7]+'m'+offsetStr, (int((5/600)*self.width), int((40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*self.width)),(255,255,255),2,cv2.LINE_AA)
             
    if plot==True:       
      cv2.imshow("Image with Curvature and Offset", imageCopy)
 
    return imageCopy
     
  def getLaneLinePreviousWindow(self, leftFit, rightFit, plot=False):
    # Use the lane line from the previous sliding window to get the parameters to fill in the lane line

    margin = self.margin
 
    # Find the x and y coordinates of all the nonzero         
    nonzero = self.warpedFrame.nonzero()  
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
         
    # Store left and right lane pixel indices
    leftLaneInds = ((nonzerox > (leftFit[0]*(nonzeroy**2) + leftFit[1]*nonzeroy + leftFit[2] - margin)) & (nonzerox < (leftFit[0]*(nonzeroy**2) + leftFit[1]*nonzeroy + leftFit[2] + margin))) 
    rightLaneInds = ((nonzerox > (rightFit[0]*(nonzeroy**2) + rightFit[1]*nonzeroy + rightFit[2] - margin)) & (nonzerox < (rightFit[0]*(nonzeroy**2) + rightFit[1]*nonzeroy + rightFit[2] + margin)))           
    self.leftLaneInds = leftLaneInds
    self.rightLaneInds = rightLaneInds

    leftx = nonzerox[leftLaneInds]
    lefty = nonzeroy[leftLaneInds] 
    rightx = nonzerox[rightLaneInds]
    righty = nonzeroy[rightLaneInds]  
 
    self.leftx = leftx
    self.rightx = rightx
    self.lefty = lefty
    self.righty = righty        
     
    leftFit = np.polyfit(lefty, leftx, 2)
    rightFit = np.polyfit(righty, rightx, 2)
    self.leftFit = leftFit
    self.rightFit = rightFit
         
    # Create the x and y values to plot on the image
    ploty = np.linspace(0, self.warpedFrame.shape[0]-1, self.warpedFrame.shape[0]) 
    leftFitx = leftFit[0]*ploty**2 + leftFit[1]*ploty + leftFit[2]
    rightFitx = rightFit[0]*ploty**2 + rightFit[1]*ploty + rightFit[2]
    self.ploty = ploty
    self.leftFitx = leftFitx
    self.rightFitx = rightFitx
         
    if plot==True:
      outImg = np.dstack((self.warpedFrame, self.warpedFrame, (self.warpedFrame)))*255
      windowImg = np.zeros_like(outImg)
             
      # Add color to the left and right line pixels
      outImg[nonzeroy[leftLaneInds], nonzerox[leftLaneInds]] = [255, 0, 0]
      outImg[nonzeroy[rightLaneInds], nonzerox[rightLaneInds]] = [0, 0, 255]
      
      # Show the search window area
      margin = self.margin
      leftLineWindow1 = np.array([np.transpose(np.vstack([leftFitx-margin, ploty]))])
      leftLineWindow2 = np.array([np.flipud(np.transpose(np.vstack([leftFitx+margin, ploty])))])
      leftLinePts = np.hstack((leftLineWindow1, leftLineWindow2))
      rightLineWindow1 = np.array([np.transpose(np.vstack([rightFitx-margin, ploty]))])
      rightLineWindow2 = np.array([np.flipud(np.transpose(np.vstack([rightFitx+margin, ploty])))])
      rightLinePts = np.hstack((rightLineWindow1, rightLineWindow2))
             
      # Draw the lane onto the warped blank image
      cv2.fillPoly(windowImg, np.int_([leftLinePts]), (0,255, 0))
      cv2.fillPoly(windowImg, np.int_([rightLinePts]), (0,255, 0))
      result = cv2.addWeighted(outImg, 1, windowImg, 0.3, 0)

      figure, (ax1, ax2, ax3) = plt.subplots(3,1) 
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.origFrame, cv2.COLOR_BGR2RGB))
      ax2.imshow(self.warpedFrame, cmap='gray')
      ax3.imshow(result)
      ax3.plot(leftFitx, ploty, color='yellow')
      ax3.plot(rightFitx, ploty, color='yellow')
      ax1.set_title("Original Frame")  
      ax2.set_title("Warped Frame")
      ax3.set_title("Warped Frame With Search Window")
      plt.show()
             
  def getLaneLineIndicesSlidingWindows(self, plot=False):
    # Get the indices of the lane line pixels using the sliding windows technique

    margin = self.margin
 
    frameSlidingWindow = self.warpedFrame.copy()
 
    # Height of the sliding windows
    windowHeight = np.int(self.warpedFrame.shape[0]/self.noOfWindows)       
 
    # Find the x and y coordinates of all the nonzero 
    nonzero = self.warpedFrame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1]) 
         
    leftLaneInds = []
    rightLaneInds = []
         
    # Current positions for pixel indices for each window,
    leftxBase, rightxBase = self.histogramPeak()
    leftxCurrent = leftxBase
    rightxCurrent = rightxBase
 
    noOfWindows = self.noOfWindows
         
    for window in range(noOfWindows):
      # Identify window boundaries in x and y and right and left
      winYLow = self.warpedFrame.shape[0] - (window + 1) * windowHeight
      winYHigh = self.warpedFrame.shape[0] - window * windowHeight
      winXleftLow = leftxCurrent - margin
      winXleftHigh = leftxCurrent + margin
      winXrightLow = rightxCurrent - margin
      winXrightHigh = rightxCurrent + margin
      cv2.rectangle(frameSlidingWindow,(winXleftLow,winYLow),(winXleftHigh,winYHigh), (255,255,255), 2)
      cv2.rectangle(frameSlidingWindow,(winXrightLow,winYLow),(winXrightHigh,winYHigh), (255,255,255), 2)
 
      # Identify the nonzero pixels in x and y within the window
      goodLeftInds = ((nonzeroy >= winYLow) & (nonzeroy < winYHigh) & (nonzerox >= winXleftLow) & (nonzerox < winXleftHigh)).nonzero()[0]
      goodRightInds = ((nonzeroy >= winYLow) & (nonzeroy < winYHigh) & (nonzerox >= winXrightLow) & (nonzerox < winXrightHigh)).nonzero()[0]

      leftLaneInds.append(goodLeftInds)
      rightLaneInds.append(goodRightInds)
         
      # Recenter next window on mean position
      minpix = self.minpix
      if len(goodLeftInds) > minpix:
        leftxCurrent = np.int(np.mean(nonzerox[goodLeftInds]))
      if len(goodRightInds) > minpix:        
        rightxCurrent = np.int(np.mean(nonzerox[goodRightInds]))
                     
    leftLaneInds = np.concatenate(leftLaneInds)
    rightLaneInds = np.concatenate(rightLaneInds)
 
    leftx = nonzerox[leftLaneInds]
    lefty = nonzeroy[leftLaneInds] 
    rightx = nonzerox[rightLaneInds] 
    righty = nonzeroy[rightLaneInds]
 
    leftFit = np.polyfit(lefty, leftx, 2)
    rightFit = np.polyfit(righty, rightx, 2) 
         
    self.leftFit = leftFit
    self.rightFit = rightFit
 
    if plot==True:
      ploty = np.linspace(0, frameSlidingWindow.shape[0]-1, frameSlidingWindow.shape[0])
      leftFitx = leftFit[0]*ploty**2 + leftFit[1]*ploty + leftFit[2]
      rightFitx = rightFit[0]*ploty**2 + rightFit[1]*ploty + rightFit[2]

      outImg = np.dstack((frameSlidingWindow, frameSlidingWindow, (frameSlidingWindow))) * 255

      # Add color to the left line pixels and right line pixels
      outImg[nonzeroy[leftLaneInds], nonzerox[leftLaneInds]] = [255, 0, 0]
      outImg[nonzeroy[rightLaneInds], nonzerox[rightLaneInds]] = [0, 0, 255]
                 
      figure, (ax1, ax2, ax3) = plt.subplots(3,1)
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.origFrame, cv2.COLOR_BGR2RGB))
      ax2.imshow(frameSlidingWindow, cmap='gray')
      ax3.imshow(outImg)
      ax3.plot(leftFitx, ploty, color='yellow')
      ax3.plot(rightFitx, ploty, color='yellow')
      ax1.set_title("Original Frame")  
      ax2.set_title("Warped Frame with Sliding Windows")
      ax3.set_title("Detected Lane Lines with Sliding Windows")
      plt.show()        
             
    return self.leftFit, self.rightFit
 
  def getLineMarkings(self, frame=None, plot=False):
    # Get lane lines

    if frame is None:
      frame = self.origFrame
             
    # Convert the video frame from BGR to HLS 

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
 
    # Perform Sobel edge detection on the lightness channel 
    _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(180, 255))
    sxbinary = edge.blurGaussian(sxbinary, ksize=3) # Reduce noise
         
    sxbinary = edge.magThresh(sxbinary, sobelKernel=3, thresh=(110, 255))
 
    # Perform binary thresholding on the saturation channel
    sChannel = hls[:, :, 2] # use only the saturation channel data
    _, sBinary = edge.threshold(sChannel, (20, 255))
     
    # Perform binary thresholding on the red channel (yellow and white contain high red values)
    _, rThresh = edge.threshold(frame[:, :, 2], thresh=(180, 255))
 
    # AND operation to reduce noise  
    rsBinary = cv2.bitwise_and(sBinary, rThresh)
 
    self.laneLineMarkings = cv2.bitwise_or(rsBinary, sxbinary.astype(np.uint8)) 

    if plot==True:
      while(1):
        cv2.imshow("Image", self.laneLineMarkings) 
        if cv2.waitKey(0):
          break
      cv2.destroyAllWindows()
        
    return self.laneLineMarkings
         
  def histogramPeak(self):
    # Get the left and right peak of the histogram

    midpoint = np.int(self.histogram.shape[0]/2)
    leftxBase = np.argmax(self.histogram[:midpoint])
    rightxBase = np.argmax(self.histogram[midpoint:]) + midpoint
 
    # (x coordinate of left peak, x coordinate of right peak)
    return leftxBase, rightxBase
         
  def overlayLaneLines(self, plot=False):
    # Overlay lane lines on the original frame

    warpZero = np.zeros_like(self.warpedFrame).astype(np.uint8)
    colorWarp = np.dstack((warpZero, warpZero, warpZero))       
         
    ptsLeft = np.array([np.transpose(np.vstack([self.leftFitx, self.ploty]))])
    ptsRight = np.array([np.flipud(np.transpose(np.vstack([self.rightFitx, self.ploty])))])
    pts = np.hstack((ptsLeft, ptsRight))
         
    # Draw lane on the warped blank image
    cv2.fillPoly(colorWarp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(colorWarp, self.invTransformationMatrix, (self.origFrame.shape[1], self.origFrame.shape[0]))
     
    # Combine the result with the original image
    result = cv2.addWeighted(self.origFrame, 1, newwarp, 0.3, 0)
         
    if plot==True:
      figure, (ax1, ax2) = plt.subplots(2,1)
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.origFrame, cv2.COLOR_BGR2RGB))
      ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
      ax1.set_title("Original Frame")  
      ax2.set_title("Original Frame With Lane Overlay")
      plt.show()   
 
    return result           
     
  def perspectiveTransform(self, frame=None, plot=False):
    # Perform the perspective transform

    if frame is None:
      frame = self.laneLineMarkings

    self.transformationMatrix = cv2.getPerspectiveTransform(self.roiPoints, self.desiredRoiPoints)
     
    self.invTransformationMatrix = cv2.getPerspectiveTransform(self.desiredRoiPoints, self.roiPoints)
 
    self.warpedFrame = cv2.warpPerspective(frame, self.transformationMatrix, self.origImageSize, flags=(cv2.INTER_LINEAR)) 
 
    # Convert image to binary
    (thresh, binaryWarped) = cv2.threshold(self.warpedFrame, 127, 255, cv2.THRESH_BINARY)           
    self.warpedFrame = binaryWarped

    # Display the perspective frame
    if plot == True:
      warpedCopy = self.warpedFrame.copy()
      warpedPlot = cv2.polylines(warpedCopy, np.int32([self.desiredRoiPoints]), True, (147,20,255), 3)
      while(1):
        cv2.imshow('Warped Image', warpedPlot)
        if cv2.waitKey(0):
          break
      cv2.destroyAllWindows()   
             
    return self.warpedFrame        
     
  def plotRoi(self, frame=None, plot=False):
    # Plot the region of interest on an image

    if plot == False:
      return
             
    if frame is None:
      frame = self.origFrame.copy()
 
    thisImage = cv2.polylines(frame, np.int32([self.roiPoints]), True, (147,20,255), 3)

    while(1):
      cv2.imshow('ROI Image', thisImage)
      if cv2.waitKey(0):
        break
    cv2.destroyAllWindows()
     
def main():
  try: 
    inputFilepath = sys.argv[1] if len(sys.argv) >= 2 else None
    outputFilepath = sys.argv[2] if len(sys.argv) >= 3 else None
    debugMode = True if len(sys.argv) == 4 and sys.argv[3] == 'True' else False 

    # Load a image
    originalFrame = cv2.imread(inputFilepath)
  
    laneObj = imageLaneDetect(origFrame=originalFrame)
  
    # Perform thresholding to get lane lines
    laneLineMarkings = laneObj.getLineMarkings(plot=debugMode)
  
    # Plot the region of interest on the image
    laneObj.plotRoi(plot=debugMode)
  
    # Perform the perspective transform to bird eye view
    warpedFrame = laneObj.perspectiveTransform(plot=debugMode)
  
    # Generate the image histogram
    histogram = laneObj.calculateHistogram(plot=debugMode)  
      
    # Find lane line pixels using the sliding window 
    leftFit, rightFit = laneObj.getLaneLineIndicesSlidingWindows(plot=debugMode)
  
    # Color in the lane line
    laneObj.getLaneLinePreviousWindow(leftFit, rightFit, plot=debugMode)
      
    # Put overlay on the original frame
    frameWithLaneLines = laneObj.overlayLaneLines(plot=debugMode)
  
    # Calculate radius of curvature
    laneObj.calculateCurvature(printToTerminal=debugMode)
  
    # Calculate center offset                                                                 
    laneObj.calculateCarPosition(printToTerminal=debugMode)
      
    # Final version of image
    frameWithLaneLines2 = laneObj.displayCurvatureOffset(frame=frameWithLaneLines, plot=True)

    # Get the filename to append it in the output filename
    filename = re.search(r'.+(\\.+)$', inputFilepath)
      
    # Create the output file
    newFilename = outputFilepath + '\\output_' + filename.group(1)[1:]  
      
    # Save the new image in the working directory
    cv2.imwrite(newFilename, frameWithLaneLines2) 

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
  except:
    print("Error")
    
if __name__ == '__main__':     
  main()