import cv2
import imutils
import scipy
import numpy as np

def convexhull(img, landmarks):
    img_ = img.copy()
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    pts = np.array(landmarks)
    convexHullPtsFace = cv2.convexHull(pts)
    cv2.polylines(img_, [convexHullPtsFace], True, (255, 255, 255), 1)
    cv2.fillConvexPoly(mask, convexHullPtsFace, 255)
    convexHullFaceImg = cv2.bitwise_and(img_, img_, mask=mask)
    return convexHullFaceImg, convexHullPtsFace

def outsidePts(img, landmarks):
    # as suggested by Prof Sanket
    convexHullFaceImg, convexHullPtsFace = convexhull(img, landmarks)
    # cv2.imshow('faceswap', convexHullFaceImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_ = img.copy()
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    maskRect = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, convexHullPtsFace, 255)

    x, y, w, h = cv2.boundingRect(convexHullPtsFace)
    xRect, yRect = np.meshgrid(range(x, x+w+1), range(y, y+h+1))
    xRect = xRect.flatten()
    yRect = yRect.flatten()
    rect = np.vstack((xRect, yRect))
    rect = rect.T
    faceCenter = (x+int(w/2), y+int(h/2))
    cv2.fillConvexPoly(maskRect, rect, 255)

    ptsOutsideFace = cv2.subtract(maskRect, mask)
    ptsOutsideFace = np.where(ptsOutsideFace==255)
    ptsOutsideFace = np.vstack((ptsOutsideFace[1], ptsOutsideFace[0]))
    ptsOutsideFace = ptsOutsideFace.T

    return ptsOutsideFace, mask, faceCenter

def rectContains(rect, point):
    if point[0]<rect[0] or point[1]<rect[1] or point[0]>rect[0]+rect[2] or point[1]>rect[1]+rect[3]:
        return False
    return True

def drawTriangle(img, trinagleList, points, face2=False):
    color = (255, 255, 255)
    points = np.array(points)
    size = img.shape
    rect = (0, 0, size[1], size[0])
    # print(len(points))
    triangle2_list = []
    for t in trinagleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            # print(pt1, pt2, pt3)
            # print(points)
            cv2.line(img, pt1, pt2, color, 1)
            cv2.line(img, pt2, pt3, color, 1)
            cv2.line(img, pt3, pt1, color, 1)
            if not face2:
                pt1_2 = np.where((points==pt1).all(axis=1))
                # print(pt1_2)
                # pry()
                pt2_2 = np.where((points==pt2).all(axis=1))
                pt3_2 = np.where((points==pt3).all(axis=1))
                triangle2_list.append([pt1_2, pt2_2, pt3_2])
    # cv2.imshow("delaunay", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img, triangle2_list


def triangulation(img_, points):
    img = img_.copy()
    size = img.shape
    rect = (0, 0, size[1], size[0])

    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))

    trinagleList = subdiv.getTriangleList()
    # print(len(trinagleList))
    # print(type(trinagleList[0]))
    img, triangle2_list = drawTriangle(img, trinagleList, points, False)
    return img, trinagleList, triangle2_list


def points(index):
    pts = []
    with open(f"/home/ubuntu/awsFaceSwap/Output/image{index}.txt") as f:
        for line in f:
            y, x = line.split(", ")
            pts.append((int(x), int(y)))
    return pts

def boundingBox(p1, p2, p3):
    x = [p1[0], p2[0], p3[0]]
    y = [p1[1], p2[1], p3[1]]

    x_topleft = np.min(x)
    y_topleft = np.min(y)
    x_botright = np.max(x)
    y_botright = np.max(y)

    xx, yy = np.meshgrid(range(int(x_topleft), int(x_botright)), range(int(y_topleft), int(y_botright)))
    xx = xx.flatten()
    yy = yy.flatten()
    ones = np.ones(xx.shape, dtype=int)

    return xx, yy, ones

def barycentric(pt1, pt2, pt3):
    bMat = np.array([[pt1[0], pt2[0], pt3[0]], [pt1[1], pt2[1], pt3[1]], [1, 1, 1]])
    xx, yy, ones = boundingBox(pt1, pt2, pt3)
    boundingboxpts = np.vstack((xx, yy, ones))
    alpha, beta, gamma = np.dot(np.linalg.pinv(bMat), boundingboxpts)
    valid_alpha = np.where(np.logical_and(alpha>-0.1, alpha<1.1))[0]
    valid_beta = np.where(np.logical_and(beta>-0.1, beta<1.1))[0]
    valid_gamma = np.where(np.logical_and(alpha+beta+gamma>-0.1, alpha+beta+gamma<1.1))[0]

    valid_al_beta = np.intersect1d(valid_alpha, valid_beta)
    inside_pts_loc = np.intersect1d(valid_al_beta, valid_gamma)
    boundingboxpts = boundingboxpts.T
    pts_in_triangle = boundingboxpts[inside_pts_loc]

    all_alpha = alpha[inside_pts_loc]
    all_beta = beta[inside_pts_loc]
    all_gamma = gamma[inside_pts_loc]

    return all_alpha, all_beta, all_gamma, pts_in_triangle


def swapFace(image, source, sourcepts, sourcetl, target, targetpts, targettl):
    img = image.copy()
    xS, yS, wS, hS = cv2.boundingRect(sourcepts)
    warpedImgS = np.zeros((img.shape), np.uint8)
    for t1, t2 in zip(sourcetl, targettl):
        pt1s = (t1[0], t1[1])
        pt2s = (t1[2], t1[3])
        pt3s = (t1[4], t1[5])

        pt1d = (t2[0], t2[1])
        pt2d = (t2[2], t2[3])
        pt3d = (t2[4], t2[5])

        alpha, beta, gamma, pts_inside = barycentric(pt1d, pt2d, pt3d)
        # print(alpha, beta, gamma)
        pts_inside = pts_inside[:, 0:2]
        # pry()
        # ignore bad triangle
        if np.shape(pts_inside)[0] == 0:
            continue

        bMat = np.array([[pt1s[0], pt2s[0], pt3s[0]], [pt1s[1], pt2s[1], pt3s[1]], [1, 1, 1]])
        bMatCoord = np.vstack((alpha, beta))
        # print(bMatCoord)
        bMatCoord = np.vstack((bMatCoord, gamma))
        warped_ptS = np.dot(bMat, bMatCoord)

        warped_ptS = warped_ptS.T
        warped_ptS[:, 0] = warped_ptS[:, 0]/warped_ptS[:, 2]
        warped_ptS[:, 1] = warped_ptS[:, 1]/warped_ptS[:, 2]
        warped_ptS = warped_ptS[:, 0:2]

        width = range(0, source.shape[1])
        height = range(0, source.shape[0])
        # print(image1.shape)
        # https://scipython.com/book/chapter-8-scipy/examples/scipyinterpolateinterp2d/
        interp1 = scipy.interpolate.interp2d(width, height, source[:, :, 0], kind='linear')
        #print(x)
        interp2 = scipy.interpolate.interp2d(width, height, source[:, :, 1], kind='linear')
        #print(len(y))
        interp3 = scipy.interpolate.interp2d(width, height, source[:, :, 2], kind='linear')
        # print(pt1d, pt2d, pt3d)
        # pry()
        for pts, x, y in zip(pts_inside, warped_ptS[:, 0], warped_ptS[:, 1]):
            x -= xS
            y -= yS
            edge1 = interp1(x, y)
            edge2 = interp2(x, y)
            edge3 = interp3(x, y)

            img[pts[1], pts[0]] = (edge1, edge2, edge3)
            warpedImgS[pts[1], pts[0]] = (edge1, edge2, edge3)

    return img, warpedImgS


def helper(target, source, title1, title2, write_path='/home/ubuntu/awsWebsite/myapp/public/images/awsOutput/'):
    '''
    target = Face to be swapped on
    source = Face being swapped
    '''
    img = target.copy()
    # main function to form delaunay triangulation
    lm1 = points(1)
    lm2 = points(2)

    xD, yD, wD, hD = cv2.boundingRect(np.asarray(lm1))
    imgFace1 = target[yD:yD+hD, xD:xD+wD]
    # cv2.imshow('face1', imgFace1)
    # cv2.waitKey(0)
    xS, yS, wS, hS = cv2.boundingRect(np.asarray(lm2))
    imgFace2 = source[yS:yS+hS, xS:xS+wS]
    # cv2.imshow('face2', imgFace2)
    # cv2.waitKey(0)swappedClone.jpg

    if len(lm1) == 0 or len(lm2) == 0:
        # print(f'No face found')
        return None
    img1, tl1, tl2_list = triangulation(target, lm1)
    lm2 = np.array(lm2)
    tl2 = []
    # thanks to mandeep for this information
    for t in tl2_list:
        p1_id, p2_id, p3_id, = t[0], t[1], t[2]
        pt1 = lm2[p1_id][0]
        pt2 = lm2[p2_id][0]
        pt3 = lm2[p3_id][0]
        tl2.append([pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1]])
    # print(len(tl2))
    # with open("/home/ubuntu/awsFaceSwap/log.log", "a") as f:
    #     f.writelines("Triangulation done")
    source_copy = source.copy()
    tFace2 = drawTriangle(source_copy, tl2, lm2, True)

    swap, warpedImg = swapFace(img, imgFace2, lm2, tl2, imgFace1, lm1, tl1)
    # cv2.imshow('swap', swap)
    # cv2.imshow('warpedImg', warpedImg)
    # cv2.waitKey(0)
    # to get only points inside of the face, set pixel values outside of the face
    # boundary to 0
    ptsOutsideFace1, mask1, face1Center = outsidePts(target, lm1)
    swap[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]] = img[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]]
    warpedImg[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]] = 0
    # with open("/home/ubuntu/awsFaceSwap/log.log", "a") as f:
    #     f.writelines("Swapping...")
    swapedClone = cv2.seamlessClone(np.uint8(swap), img, mask1, face1Center, cv2.MIXED_CLONE)
    random_inte = np.random.randint(0,5000)
    # print(f"{write_path}{title1}{random_inte}{title2}")
    cv2.imwrite(f"{write_path}{title1}{random_inte}{title2}", swapedClone)
    # cv2.imshow('faceswap', swapedClone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"images/awsOutput/{title1}{random_inte}{title2}", end="")
