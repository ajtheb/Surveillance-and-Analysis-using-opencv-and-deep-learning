from cv2 import *
import cv2
import sys
import dlib

#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('C:\\Users\\Apoorv Jain\\Anaconda3\\envs\\env_dlib\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture('C:\\Users\\Apoorv Jain\\Documents\\deep-learning-face-detection\\sample2.mp4')
trackers = []
trackingFace = 1
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not trackingFace:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        trackingFace=1
        t = dlib.correlation_tracker()
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rect = dlib.rectangle(x, y, x + w, y + h)
            t.start_track(gray, rect)
            trackers.append(t)

    else:
        print("tracking", len(trackers))
        # loop over each of the trackers
        if (len(trackers) == 0):
            trackingFace = 0
        for t in trackers:
            # update the tracker and grab the position of the tracked
            # object
            trackingQuality = t.update(frame)
            print("tracking quality", trackingQuality)
            if trackingQuality >= 5.50:
                pos = t.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                l="person"
                # draw the bounding box from the correlation object tracker
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 0, 0), 2)
                cv2.putText(frame, l, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            else:
                trackers.remove(t)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()