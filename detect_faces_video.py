#!/usr/bin/python
'''
	Author: Guido Diepen <gdiepen@deloitte.nl>
'''

#Import the OpenCV and dlib libraries
import cv2
import dlib


#Initialize a face cascade using the frontal face haar cascade provided with
#the OpenCV library
faceCascade = cv2.CascadeClassifier('C:\\Users\\Apoorv Jain\\Anaconda3\\envs\\env_dlib\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600

def detectAndTrackLargestFace():
	#Open the first webcame device
	capture = cv2.VideoCapture('C:\\Users\\Apoorv Jain\\Videos\\sample1.mp4')
	faceCount=0

	#Create two opencv named windows
	cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
	cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

	#Position the windows next to eachother
	cv2.moveWindow("base-image",0,100)
	cv2.moveWindow("result-image",400,100)

	#Start the window thread for the two windows we are using
	cv2.startWindowThread()

	#Create the tracker we will use


	#The variable we use to keep track of the fact whether we are
	#currently using the dlib tracker
	trackingFace = 0

	#The color of the rectangle we draw around the face
	rectangleColor = (0,165,255)
	tracker = dlib.correlation_tracker()

	try:
		while True:
			#Retrieve the latest image from the webcam
			rc,fullSizeBaseImage = capture.read()

			#Resize the image to 320x240
			baseImage = cv2.resize( fullSizeBaseImage, ( 320, 240))


			#Check if a key was pressed and if it was Q, then destroy all
			#opencv windows and exit the application
			pressedKey = cv2.waitKey(2)
			if pressedKey == ord('q'):
				cv2.destroyAllWindows()
				exit(0)





			#Result image is the image we will show the user, which is a
			#combination of the original image from the webcam and the
			#overlayed rectangle for the largest face
			resultImage = baseImage.copy()

			gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
			# Now use the haar cascade detector to find all faces
			# in the image
			faces = faceCascade.detectMultiScale(gray, 1.3, 5)
			#print("number", len(faces))
			trackerlist=[]



			#If we are not tracking a face, then try to detect one
			if not trackingFace:

				#For the face detection, we need to make use of a gray
				#colored image so we will convert the baseImage to a
				#gray-based image
				gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
				#Now use the haar cascade detector to find all faces
				#in the image
				faces = faceCascade.detectMultiScale(gray, 1.3, 5)
				#print("number",len(faces))
				#In the console we can show that only now we are
				#using the detector for a face
				#print("Using the cascade detector to detect face")


				#For now, we are only interested in the 'largest'
				#face, and we determine this based on the largest
				#area of the found rectangle. First initialize the
				#required variables to 0
				maxArea = 0
				x = 0
				y = 0
				w = 0
				h = 0


				#Loop over all faces and check if the area for this
				#face is the largest so far
				#We need to convert it to int here because of the
				#requirement of the dlib tracker. If we omit the cast to
				#int here, you will get cast errors since the detector
				#returns numpy.int32 and the tracker requires an int
				#print("face",len(faces))

				for (_x,_y,_w,_h) in faces:
					x = int(_x)
					y = int(_y)
					w = int(_w)
					h = int(_h)


					#If one or more faces are found, initialize the tracker
					#on the largest face in the picture
					faceCount=faceCount+1

					cv2.rectangle(resultImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
					print(x,y)
					#trackerlist.append(tracker)
					#Set the indicator variable such that we know the
					#tracker is tracking a region in the image
			#Check if the tracker is actively tracking a region in the image






			#Since we want to show something larger on the screen than the
			#original 320x240, we resize the image again
			#
			#Note that it would also be possible to keep the large version
			#of the baseimage and make the result image a copy of this large
			#base image and use the scaling factor to draw the rectangle
			#at the right coordinates.
			largeResult = cv2.resize(resultImage,
									 (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

			#Finally, we want to show the images on the screen
			cv2.imshow("base-image", baseImage)
			cv2.imshow("result-image", largeResult)
			print(faceCount)



	#To ensure we can also deal with the user pressing Ctrl-C in the console
	#we have to check for the KeyboardInterrupt exception and destroy
	#all opencv windows and exit the application
	except KeyboardInterrupt as e:
		cv2.destroyAllWindows()
		exit(0)


if __name__ == '__main__':
	detectAndTrackLargestFace()