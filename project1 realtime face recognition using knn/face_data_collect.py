import cv2
import numpy as np

cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
face_data=[]

dataset_path='./data/'
file_name=input("enter the name of person whose face we are scanning")


while True:
	ret,frame =cap.read()

	if ret==False:
		continue

	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	#print("before")
	#print(faces)
	#print("after")
	# print(len(faces))
	# for i in (len(faces)):
	# 	print(faces[i])
	# 	print("in")
	# print("out")

	if len(faces)==0:
		continue
	faces = sorted(faces,key=lambda f:f[2]*f[3])


	#sorting the faces in a frame.i.e to get larger face.
	#sorting will be done on the basis of area i.e product of width and height
	#face[2]*faces[3]

	#because last face is largest face.
	for (x,y,w,h) in faces[-1:]:
		#print("in for")
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crp out the required face) : Region of interest
		#padding aound fcae in a extracted box of fcae		
		offset=10# in pixels.

		#in frame by convention ->frame[Y,X]
		face_section= frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))# can be any size
		
		skip+=1
		#store every  10th face
		if skip%10 == 0:
			face_data.append(face_section)
			print(len(face_data))		

	cv2.imshow("frame",frame)
	cv2.imshow("face_section",face_section)

	key_pressed=cv2.waitKey(1) & 0xFF

	#ord function tells ascii value of character
	if key_pressed == ord('q'):
		break

#convert face list array into a numpy array
face_data = np.asarray(face_data)
#print(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
#print("now")
#print(face_data)

#saving data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("data saved successfully")

cap.release()
cv2.destroyAllWindows()