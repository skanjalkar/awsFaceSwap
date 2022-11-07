import dlib
import cv2
import imutils
import numpy as np

# https://www.studytonight.com/post/dlib-68-points-face-landmark-detection-with-opencv-and-python

# This below mehtod will draw all those points which are from 0 to 67 on face one by one.
def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False):
  points = []
  for i in range(startpoint, endpoint+1):
    point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
    points.append(point)

  points = np.array(points, dtype=np.int32)
  cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

# Use this function for 70-points facial landmark detector model
# we are checking if points are exactly equal to 68, then we draw all those points on face one by one
def facePoints(image, faceLandmarks):
    assert(faceLandmarks.num_parts == 68)
    drawPoints(image, faceLandmarks, 0, 16)           # Jaw line
    drawPoints(image, faceLandmarks, 17, 21)          # Left eyebrow
    drawPoints(image, faceLandmarks, 22, 26)          # Right eyebrow
    drawPoints(image, faceLandmarks, 27, 30)          # Nose bridge
    drawPoints(image, faceLandmarks, 30, 35, True)    # Lower nose
    drawPoints(image, faceLandmarks, 36, 41, True)    # Left eye
    drawPoints(image, faceLandmarks, 42, 47, True)    # Right Eye
    drawPoints(image, faceLandmarks, 48, 59, True)    # Outer lip
    drawPoints(image, faceLandmarks, 60, 67, True)    # Inner lip

# Use this function for any model other than
# 70 points facial_landmark detector model
def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
  for p in faceLandmarks.parts():
    cv2.circle(image, (p.x, p.y), radius, color, -1)

def storeLandmarks(faceLandmarks, fileName):
    with open (fileName, "w") as f:
        for point in faceLandmarks.parts():
            f.write(f'{int(point.y)}, {int(point.x)}\n')
    f.close()


def landmark(plot, image1_path, image2_path, path="/home/ubuntu/awsFaceSwap/shape_predictor_68_face_landmarks.dat", op_path="/home/ubuntu/awsFaceSwap/"):
    # with open("/home/ubuntu/awsFaceSwap/log.log", "a") as f:
    #     f.writelines("Landmarks before dlib\n")
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(path)
    images_dict = {1: image1_path, 2: image2_path}
    image_paths = []
    images  = []
    # with open("/home/ubuntu/awsFaceSwap/log.log", "a") as f:
        # f.writelines("Landmarks starting\n")
    for index in range(1, 3):
        image = images_dict[index]
        img = cv2.imread(image)
        img = imutils.resize(img, width=320)
        img_ = img.copy()
        images.append(img_)
        # with open("/home/ubuntu/awsFaceSwap/log.log", "a") as f:
        #     f.writelines("Image read\n")
        # cv2.imshow("Original Image", img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.waitKey(1000)
        faceLandmarkOp = f"Output/image"
        all_faces = face_detector(img, 1)

        allLandmarks = []

        # print("List of all faces detected: ",len(all_faces))

        for k in range(len(all_faces)):
            rect = all_faces[k]
            # print(rect)
            faceRect = dlib.rectangle(int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))
            # print(faceRect)
            cv2.rectangle(img, (int(rect.left()), int(rect.top())), (int(rect.right()), int(rect.bottom())), (0, 0, 255), 2)
            detected = landmark_detector(img, faceRect)
            # if k == 0:
                # print("Total number of face landmarks detected ",len(detected.parts()))
            allLandmarks.append(detected)
            facePoints(img, detected)
            fileName = op_path + "Output/image" + f"{index}.txt"
            # print(f'Landmark is saved into {fileName}')


            storeLandmarks(detected, fileName)
        # with open("/home/ubuntu/awsFaceSwap/log.log", "a") as f:
        #     f.writelines("Landmarks done")

        opimg = op_path + f'Output/result{index}.jpg'
        image_paths.append(opimg)
        cv2.imwrite(opimg, img)
        # if plot:
        #     cv2.imshow("Result", img)
        #     cv2.waitKey(5000)
        #     cv2.destroyAllWindows()
    return image_paths, images