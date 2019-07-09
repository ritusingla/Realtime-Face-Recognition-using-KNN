# Read video stream from webcam (frame by frame)
import cv2

#step1 - 
#to capture a device from which youw want to read the video stream.

#ByDefault the id of our webcam is 0
# if we have multiple webcams then we can trey with mutiple ids
cap=cv2.VideoCapture(0)
#object of haarcascades classifier which works on facial data(this classifier is about face detection only)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# returns 2 things:
	#1.Boolean value(return value)
	# if it is false , it means frame is not captured properly.
    #2. Frame that has been captured
	if ret == False:
		continue

	#we are going to fit this image(frame) onto object
	#scalingFactor=1.3 and NoOfneighbours=5
	faces=face_cascade.detectMultiScale(frame,1.3,5)


	cv2.imshow("video Frame",frame)
	#cv2.imshow("gray video Frame",gray_frame)

	#we are going to loop over all the faces return by detectMultiScale function and drawing a rounding box arnd each face.

	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow("video Frame",frame)


	#wait for user input- q(entered by user), then this loop will stop
	#we need to check what key is pressed by user.

	#it means program will wait for 1ms before next iteration comes up
	key_pressed=cv2.waitKey(1) & 0xFF

	#ord function tells ascii value of character
	if key_pressed == ord('q'):
		break
cap.release()#releasing the device
cv2.destroyAllWindows()