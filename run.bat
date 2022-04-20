@echo Is the input (image/video)?
@set /p filetype=
@echo Enter the input image/video path (without quotation marks):
@set /p inputpath= 
@echo Enter the image/video path to be saved at (without quotation marks):
@set /p outputpath=
@echo Enter debug mode value (True/False) (without quotation marks):
@set /p debugMode=

@if /I "%filetype%"=="image" python imageLaneDetect.py "%inputpath%" "%outputpath%" "%debugMode%"
@if /I "%filetype%"=="video" python videoLaneDetect.py "%inputpath%" "%outputpath%" "%debugMode%"
@pause
