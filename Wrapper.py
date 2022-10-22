#from tkinter.tix import Tree
from Functions import landmarks
from Functions import triangle
import argparse
import cv2
#import pry

if __name__ == "__main__":
    Parser  = argparse.ArgumentParser()
    Parser.add_argument('--plot',default=True, help='Save landmarks')
    image_paths, images = landmarks.landmark(True)
    # pry()
    # for path in image_paths:
    #     img = cv2.imread(path)
    #     images.append(img)
    triangle.helper(images[0], images[1])
    # triangle.helper(images[0], images[1])
    #https://github.com/davisking/dlib-models
