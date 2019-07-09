import cv2

img=cv2.imread('dog.png')
gray=cv2.imread('dog.png',cv2.IMREAD_GRAYSCALE)

cv2.imshow('A Dog',img)
cv2.imshow('GRAY Dog',gray)
#This function basically shows the time 
#0 means infinite time if suppose it was 25 then it means wait for 25ms before the window gets distriyed.
cv2.waitKey(0)

#window will not destroy of its own. beacuase waitkey is set to 0
cv2.destroyAllWindows()
