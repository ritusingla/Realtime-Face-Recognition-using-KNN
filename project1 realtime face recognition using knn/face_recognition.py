""" 
Recognising faces using some classification algorithm.

1.Read a video stream 
2.extract files out of it
3. load the training data (numpy arrays of all the persons.)
	x-values are stored in numpy arrays.
	y-values we need to assign to each person.
4. use knn to find the pediction of face(int)
5.map the predicted id to name of the user
6.Display the predictions on the screen-bounding box and name


"""

import cv2
import numpy as np
import os

"""      KNN Code   """
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    #it will store the distance
    vals=[]
    #total number of points
    m=X.shape[0]
    
    for i in range(m):
        d=dist(queryPoint,X[i])
        vals.append((d,Y[i]))
    
    #sorting is done on the basis of first parameter.
    vals=sorted(vals)
    #Nearest/first K points
    vals=vals[:k]
    
    vals=np.array(vals)
    #print(vals)
    #it returns how many unique points are there alongwith their count
    # we are calling this function only on Y values or labels i.e. 2nd column of vals.
    #in o/p -> first part gives labels and second part their count
    new_vals=np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    
    #here we get max count index.
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    return pred
    

##finished knn code

cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0

#x values of data
face_data=[]
#y values
labels=[]
dataset_path='./data/'

#for files that we are going to load form data folders
class_id=0#label for given file

names={}#mapping b/w id-name

#data preparation
#listdir function gives all the files in given path/folder
for file in os.listdir(dataset_path):
	if file.endswith('.npy'):
		names[class_id]=file[:-4]
		print(file)
		data_item=np.load(dataset_path+file)
		face_data.append(data_item)

		#create labels
		label_ =class_id*np.ones((data_item.shape[0]))
		print("label shape")
		print(label_.shape)
		class_id+=1
		labels.append(label_)

#see notebook for once
face_dataset=np.concatenate(face_data,axis=0)# reshaping it is same as flatten() function that we used below for the captured image in current window
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)



#Now we want to combine both X and Y matrices because our knn algorithm takes train data in which last column represent labels and rest beginning columns represent x values

# train=np.concatenate((face_dataset,face_labels),axis=1)
# print(train.shape)


##Do testing

#read frames .
while True:
	ret,frame = cap.read()
	#gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	if ret == False:
		continue

	faces=face_cascade.detectMultiScale(frame,1.3,5)

	#extraction of face.
	for(x,y,w,h) in faces:
		offset=10# in pixels.
		#in frame by convention ->frame[Y,X]
		face_section=frame[y-offset:y+h+offset,x-offset : x+w+offset]
		face_section=cv2.resize(face_section,(100,100))# can be any size

		out=knn(face_dataset,face_labels,face_section.flatten())

		#display on screen the name and rectangle arond it.
		cv2.putText(frame,names[int(out)],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)


	
	key_pressed=cv2.waitKey(1) & 0xFF
	cv2.imshow("faces",frame)
	#ord function tells ascii value of character
	if key_pressed == ord('q'):
		break
cap.release()#releasing the device
cv2.destroyAllWindows()



"""
		LIMITATION:
		for the case when new person face has to be detectedd which is not present in the database.
		It shows one of the name from the database which resembles the most.
		OR
		IN OTHER WORDS
		It cannot detect the new faces which are not present in database.(humans only)


"""