# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt",
                default='C:\\Users\\Apoorv Jain\\Documents\\multiobject-tracking-dlib\\deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model",
                default='C:\\Users\\Apoorv Jain\\Documents\\multiobject-tracking-dlib\\res10_300x300_ssd_iter_140000.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", default='C:\\Users\\Apoorv Jain\\Videos\\samples\\sample6.mp4',
                help="path to input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# load our serialized model from disk
print("[INFO] loading models (face detection and mask detection)...")

# face detection
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# mask detection
mask_model= "C:\\Users\\Apoorv Jain\\Documents\\GitHub\\Surveillance-and-Analysis-using-Image-Processing\\model\\mask_model.model"
maskNet = load_model(mask_model)



# initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

# saving output in video
fr_width=int(vs.get(3))
fr_height=int(vs.get(4))
fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
fsize=(fr_width,fr_height)

result = cv2.VideoWriter('output3.avi', fourcc, 20.0, (600,math.floor((600/fr_width)*fr_height)))

# initialize the list of object trackers and corresponding class and labels
trackers = []
labels = []

# count of faces,mask,no mask
faceCount = 0
mask_cnt=0
no_mask_cnt=0

# start the frames per second throughput estimator
fps = FPS().start()

# 0 means not tracking now and 1 means tracking now
trackingFace = 0

# timer of tracking the same list of faces
epoch_counter = 0

# number of faces in previous frame
prev=0
preds = []

# loop over frames from the video file stream
while True:
    # grab the next frame from the video file
    (grabbed, frame) = vs.read()

    # check to see if we have reached the end of the video file
    if frame is None:
        break

    # resize the frame for faster processing and then convert the
    # frame from BGR to RGB ordering (dlib needs RGB ordering)
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if there are no object trackers we first need to detect objects
    # and then create a tracker for each object
    if not trackingFace:
        (h, w) = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        #print("size", detections.shape[2])

        c = 0
        # Calculating face Count
        # Start frame is not sufficient to make correct prediction,so third frame is used
        if epoch_counter>2:
            faceCount-=prev

        # no face
        if(detections.shape[2]==0):
            continue

        epoch_counter += 1

        # loop over the detections
        for i in range(0, detections.shape[2]):

            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum confidence
            if confidence > args["confidence"]:
                c = c + 1

                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if frame is None:
                    continue

                if grabbed == False:
                    continue

                # preprocessing for mask prediction--
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces=[]
                faces.append(face)
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)
                (mask, withoutMask) = preds[0]
                label = "Mask" if mask > withoutMask else "No Mask"


                if label == 'Mask':
                    mask_cnt+=1
                else:
                    no_mask_cnt+=1

                color = (0, 255, 0) if label == "Mask" else (0,0,255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # draw bounding box and label on the face
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              color, 2)

                # incrementing face Count
                if epoch_counter>2:
                    faceCount+=1


                if(epoch_counter>2):
                    # construct a dlib rectangle object from the bounding
                    # box coordinates and start the correlation tracker
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    t.start_track(rgb, rect)

                    # update our set of trackers and corresponding class
                    # labels
                    labels.append(label)
                    trackers.append(t)
                    trackingFace = 1


    # otherwise, we've already performed detection so let's track
    # multiple objects
    else:
        #print("hello")
        # if we are tracking for too long and search faces again
        if(epoch_counter>220):
            epoch_counter=0
            prev=len(trackers)
            trackers.clear()
            trackingFace=0

        # if the person go out of the frame
        if (len(trackers) == 0):
            prev=0
            epoch_counter=0
            trackingFace = 0

        # loop over each of the trackers
        for (t, l) in zip(trackers, labels):
            # update the tracker and grab the position of the tracked
            # object
            trackingQuality = t.update(rgb)

            if trackingQuality >= 5.50:
                pos = t.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # draw the bounding box from the correlation object tracker
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              color, 2)
                cv2.putText(frame, l, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            else:
                trackers.remove(t)
                labels.remove(l)

        epoch_counter+=1

    print("faceCount till now : ", faceCount)

    # show the output frame
    cv2.imshow("Frame", frame)
    result.write(frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

#final Output
print("Report---")
print("Number of people entered :",faceCount)
print("Number of people with mask :",mask_cnt)
print("Number of people without mask :",no_mask_cnt)

# stop the timer and display FPS information
fps.stop()

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
result.release()