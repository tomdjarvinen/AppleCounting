import numpy as np
import cv2
import queue
import csv
class Apple:
    #Handles information regarding state of detected apples in system including updating position, and modeling motion
    #Will use Kalman filter model
    def __init__(self, position,confidence, prior = [0.0,0.0]):
        # Position refers to the x,y position of the apple in the current active image
        self.position = position
        # These members define the current motion model of the apple as x,y vectors.  Unit of velocity is pixel/frame
        #TODO: Implement and test motion model
        self.velocity = prior
        # Size defines the largest detected size of the currently detected apple in terms of major and minor axes (mm)
        self.size = self.estimateSize(position)
        # Value varying from 0-1; output from object detector
        self.confidence = confidence
        #boolean value indicating whether the apple has left the edge of the images
        self.inImage = 1
        #track the number of times
        self.numDetections = 1

    def estimateSize():
        #TODO: Implement size estimation
        return 0
    def translate():
        position += velocity
    def sensorInput():
        pass

from abc import ABC, abstractmethod
class Video(ABC):
    #Handler for processing of video frames
    def __init__(self, imageSource):
        self.currentFrame = 0
    @abstractmethod
    def detectApples(frame):
        pass
    @abstractmethod
    def computeFlow():
        pass
    @abstractmethod
    def advanceFrame():
        pass



class FrameList(Video):
    #implementation of video handler which takes a list of PNG filenames and precomputed detection results.
    #Inputs:
        #imageSource: ordered list of image filenames; expected to be at least 3 frames
        #detectionResults:  text file; space delimited; fields in each line are: imageName confidence xmin ymin xmax ymax
        #threshold: defines minimum confidence value for a detection to be considered valid,
            #this value can be tuned to reduce either false positives or false negatives
    def __init__(self,imageSource,detectionResults,threshold = 0.7):
        #populate detection List
        #Detections are unsorted, so we first create a dictionary of detection List
        detections = self._populateDetections(detectionResults,threshold)
        #next we create a sorted list of Detections
        self._detections = list()
        for i in imageSource:
            self._detections.append(detections[i])
        #Populate image queue
        if len(imageSource) >=2:
            self.frameOld = cv2.imread(imageSource[0] + '.png')
            self.frameNew = cv2.imread(imageSource[1] + '.png')
            self.currentFrame = 1
            self.ImageList = imageSource
        else:
            raise Exception('image list should be at least 2 frames')

    def _populateDetections(self,detectionFile, threshold):
        #Read detection file
        with open(detectionFile) as csvfile:
            reader = csv.reader(csvfile, delimiter = ',')
            dataList = []
            for row in reader:
                dataList.append(row)
        w, h = 6,len(dataList)
        AnnotationMatrix = [[0 for x in range(w)] for y in range(h)]
        for index, i in enumerate(dataList):
            tempArray = i[0].split()
            for index2,j in enumerate(tempArray):
                if index2 == 0:
                    AnnotationMatrix[index][index2] = j
                else:
                    AnnotationMatrix[index][index2] = float(j)
        confidence = 1.0
        iterator = 0
        annotations = list()
        detections = dict()
        while iterator < (len(dataList)-1):
            startName = AnnotationMatrix[iterator][0]
            #Parse through annotations of first image
            filename = AnnotationMatrix[iterator][0]
            detection = list()
            while filename == startName and iterator < (len(dataList)-1):
                confidence = AnnotationMatrix[iterator][1]
                if confidence >= threshold:
                    detection.append([int(AnnotationMatrix[iterator][2]),int(AnnotationMatrix[iterator][3]),
                                        int(AnnotationMatrix[iterator][4]),int(AnnotationMatrix[iterator][5])])
                iterator = iterator + 1
                filename = AnnotationMatrix[iterator][0]
            detections[startName] = detection
        return detections
    def detectApples(self):
        #detects apples in the current frame
        if(self.currentFrame >= 0):
            return self._detections[self.currentFrame]
        else:
            return None
    def computeFlow(self, apples, paramDict = None):
        #use detection results with Lucas Kanade Optical flow operator
        if paramDict is None:
            lk_params = dict(winSize  = (30,30),
                            maxLevel = 5,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))
        else:
            lk_params = paramDict
        #Get features to track
        #TODO: This compution should not be performed here; feature selection should be performed in appleSet Class
        centroids = np.ndarray(shape = (len(apples),1,2),dtype = np.float32())
        for i, value in enumerate(apples):
                x = (value[0]+value[2])/2
                y = (value[1]+value[3])/2
                centroids[i,0,:] = [x,y]
        old_gray = cv2.cvtColor(self.frameOld, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(self.frameNew, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, centroids, None, **lk_params)
        return p1
    def advanceFrame(self):
        self.currentFrame += 1
        self.frameOld = self.frameNew
        self.frameNew = cv2.imread(self.ImageList[self.currentFrame] + '.png')
    def displayCurrentFrame(self, detections = None, trajectories = None,color = (0,255,0),linewidth = 4):
        #Helper function to display frameNew
        #Inputs: detections: list of apple bounding boxes in format [xmin,ymin,xmax,ymax]
        #TODO: This needs to be modified to work nicely with however I decide to format data in apple class.
        frame = self.frameNew
        if detections is not None:
            for i in detections:
                frame = cv2.rectangle(frame,(i[0],i[1]),(i[2],i[3]),color,thickness = linewidth)
        if trajectories is not None:
                for i in trajectories:
                    frame = cv2.line(frame,i[0],i[1],5,i[2],3)
                    frame = cv2.circle(frame,i[0],5,i[2],3)
        cv2.imshow('frame',frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
class appleSet:
    #Handler for a  partition of apple fruit counts.
    def __init__(self,videoHandler):
        self.vid = videoHandler
        self.apples = []
