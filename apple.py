import numpy as np
class Apple:
    def __init__(self, position,confidence):
        # Position refers to the x,y position of the apple in the current active image
        self.position = position
        # These members define the current motion model of the apple as x,y vectors.  Unit of velocity is pixel/frame
        #TODO: Implement and test motion model
        self.velocity = [0.0,0.0]
        self.acceleration = [0.0,0.0]
        # Size defines the largest detected size of the currently detected apple in terms of major and minor axes (mm)
        self.size = self.estimateSize(position)
        # Value varying from 0-1; output from object detector
        self.confidence = confidence
        #boolean value indicating whether the apple has left the edge of the images
        self.inImage = 1
    def estimateSize():
        #TODO: Implement size estimation
        return 0
    def translate(flow):
        #TODO: Implement
        #Translate position of apple based on optical flow
        translation = median(flow[self.position[1]:self.position[3],self.position[2]:self.position[4]])
        self.position += translation
        self.accleration = translation - self.velocity
        self.velocity = translation

    
