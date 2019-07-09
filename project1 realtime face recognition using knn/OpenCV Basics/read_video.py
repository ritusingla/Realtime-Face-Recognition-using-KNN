# Read video stream from webcam (frame by frame)
import cv2

#step1 - 
#to capture a device from which youw want to read the video stream.

#ByDefault the id of our webcam is 0
# if we have multiple webcams then we can trey with mutiple ids
cap=cv2.VideoCapture(0)

while True:
	ret,frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# returns 2 things:
	#1.Boolean value(return value)
	# if it is false , it means frame is not captured properly.
    #2. Frame that has been captured
	if ret == False:
		continue

	cv2.imshow("video Frame",frame)
	cv2.imshow("gray video Frame",gray_frame)
	#wait for user input- q(entered by user), then this loop will stop
	#we need to check what key is pressed by user.

	#it means program will wait for 1ms before next iteration comes up
	key_pressed=cv2.waitKey(1) & 0xFF

	#ord function tells ascii value of character
	if key_pressed == ord('q'):
		break
cap.release()#releasing the device
cv2.destroyAllWindows()