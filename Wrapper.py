#from tkinter.tix import Tree
from Functions import landmarks
from Functions import triangle
import cv2
import sys
#import pry

if __name__ == "__main__":
    global_path = "/home/ubuntu/awsWebsite/myapp/"
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    orignal_name1 = sys.argv[3]
    orignal_name2 = sys.argv[4]
    # with open("/home/ubuntu/awsFaceSwap/log.log", "a") as f:
    #     f.writelines("Read the inputs\n")
    #     f.writelines(global_path+image1_path+"\n")
    #     f.writelines(global_path+image2_path+"\n")
    image_paths, images = landmarks.landmark(False, global_path+image1_path, global_path+image2_path)
    # pry()
    # for path in image_paths:
    #     img = cv2.imread(path)
    #     images.append(img)
    triangle.helper(images[0], images[1], orignal_name1, orignal_name2)
    # triangle.helper(images[0], images[1])
    #https://github.com/davisking/dlib-models
