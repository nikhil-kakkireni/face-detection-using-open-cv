import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img=cv2.imread("family.jpg")#using grayscale increases accuracy in detection
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cap  = cv2.VideoCapture(0)
#above line captures frames from camera
#we have to read the captured frame using cap.read()

#scale factor :-smaller the value higher the accuracy
faces=face_cascade.detectMultiScale(gray_img,
                                    scaleFactor=1.1,
                                    minNeighbors=5)
for x, y, w, h in faces:
    img=cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)


print(type(faces))#returns numpy array
print(faces)#returns startting point(indexes(x,y)) of face,width and height of the face

resized=cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))

cv2.imshow("gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()