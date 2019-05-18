import json
import numpy as np
import dlib
import cv2
import os
import random

_DLIB_FACE_DETECTOR = dlib.get_frontal_face_detector()


def load_landmarker_data(json_path):
    """
    parse a pair of annotation/image files
    :param json_path: path to the json annotation
    :return: a tuple containing the image in HxWx(RGB) format and
                the keypoints in Nx(RC) format
    """
    with open(json_path, mode='r') as f:
        loaded = json.load(f)

    points = loaded['landmarks']['points']
    if None in points:
        raise ValueError()

    # note: points are in row column, not X, Y
    points = np.array(points)

    img = dlib.load_rgb_image(json_path.split('_ibug')[0])

    height, width, _ = img.shape
    points = np.round(points)
    points[:, 0] = np.clip(points[:, 0], 0, height)
    points[:, 1] = np.clip(points[:, 1], 0, width)
    return img, points.astype(np.int32)


def get_face_bb(img, points):
    """
    Get a bounding box that merges the results of the dlib face detector and
        the implied bounding box from keypoint information
    :param img: The image in HxWx(RGB) format
    :param points: The key points in Nx(RC) format
    :return: a tuple containing:
                the upper left bounding box coordinate in RC order
                the lower right bounding box coordinate in RC order
                a flag which indicates that the bounding box is only based on
                    keypoints because the detector could not find a corresponding box
                    to merge
    """

    implied_bb_ul = np.min(points, axis=0)
    implied_bb_lr = np.max(points, axis=0)
    implied_bb_area = np.prod(implied_bb_lr - implied_bb_ul)

    detections = _DLIB_FACE_DETECTOR(img, 1)
    best_detected_bb_ul = None
    best_detected_bb_lr = None
    best_iou = 0
    for face_n, bbox in enumerate(detections):
        detected_bb_ul = np.array([bbox.top(), bbox.left()])
        detected_bb_lr = np.array([bbox.bottom(), bbox.right()])
        detected_bb_area = np.prod(detected_bb_lr - detected_bb_ul)

        intersection_ul = np.maximum(detected_bb_ul, implied_bb_ul)
        intersection_lr = np.minimum(detected_bb_lr, implied_bb_lr)
        intersection_area = np.prod(intersection_lr - intersection_ul)

        cur_iou = intersection_area / (detected_bb_area + implied_bb_area - intersection_area)

        if cur_iou > best_iou:
            best_iou = cur_iou
            best_detected_bb_ul = detected_bb_ul
            best_detected_bb_lr = detected_bb_lr

    if best_iou < 0.5:
        bb_ul = implied_bb_ul
        bb_lr = implied_bb_lr
        is_implied = True
    else:
        bb_ul = np.minimum(best_detected_bb_ul, implied_bb_ul)
        bb_lr = np.maximum(best_detected_bb_lr, implied_bb_lr)
        is_implied = False

    # display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.rectangle(
    #     display,
    #     (implied_bb_ul[1], implied_bb_ul[0]),
    #     (implied_bb_lr[1], implied_bb_lr[0]), (255, 0, 0), 5)
    # cv2.rectangle(
    #     display,
    #     (bb_ul[1], bb_ul[0]),
    #     (bb_lr[1], bb_lr[0]), (0, 255, 0), 5)
    # #print("box", bb_ul, bb_lr)
    #
    # for point in points:
    #     y, x = point
    #     cv2.circle(display, (x, y), 3, (0, 255, 255), -1)
    #
    # display = cv2.resize(display, (0, 0), fx=0.3, fy=0.3)
    # cv2.imshow('d', display)
    # cv2.waitKey()

    return bb_ul, bb_lr, is_implied


def drawPoly(frame, newpoints, new_full_filename, markPoints, bbox_pt1, bbox_pt2, colorFactor=0):
    """
    write out image with points and numbers
    :param frame: cv2 image array
    :param pts: feature points np.array with shape (numPoint, 2)
    :param file_path: full file path
    :param folder: folder name
    :param type: type of augmentation
    :param angle: angle of rotation
    :param markPoints(bool): whether to draw numbers
    :param bbox_pt1: upper left point in HxWx(RGB) tuple format
    :param bbox_pt2: lower right point in HxWx(RGB) tuple format
    :param colorFactor: helps flip between red and blue, 0 -> red, 1 -> blue
    :return:
    """
    for i in range(len(newpoints)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        #origin = ([np.array(newpoints)][0][i][0], [np.array(newpoints)][0][i][1])
        origin = (newpoints[i][0], newpoints[i][1])
        if markPoints == True:
            cv2.putText(frame, str(i), origin, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, origin, 2, color=(255 * (1-colorFactor), 0, 255 * colorFactor, 255), thickness=3)
            cv2.rectangle(frame, bbox_pt1, bbox_pt2, (255, 255, 255), 2)

    cv2.imwrite(new_full_filename, frame)
    return frame

def lm_bbox_to_cvpoints(bb_ul, bb_lr):
    """
    convert landmarker point format to cv2 format
    :param bb_ul: bb upper left in Nx(RC) format
    :param bb_lr: bb lower right in Nx(RC) format
    :return: tuple containing:
        upper left but in HxWx(RGB) tuple format
        lower right but in HxWx(RGB) tuple format
    """
    bbox_pt1 = (bb_ul[1], bb_ul[0])
    bbox_pt2 = (bb_lr[1], bb_lr[0])
    return bbox_pt1, bbox_pt2

def getShapePoints(img_path, shapePredictor):
    '''
    This section deals with the inferenced points after model is trained
    '''
    DLIB_FACE_DETECTOR = dlib.get_frontal_face_detector()
    #DLIB_FACE_DETECTOR = dlib.fhog_object_detector("lioneldetectorv1_05_10.svm")  # testing lionel detector
    DLIB_SHAPE_PREDICTOR = dlib.shape_predictor(shapePredictor)
    # Get the PNG to work with

    #img = dlib.load_rgb_image(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # Use dlib to get a bounding bos and then the face shape points
    bb = DLIB_FACE_DETECTOR(img, 1)
    projectedPtsIter = DLIB_SHAPE_PREDICTOR(img, bb[0])
    # map points interator from dlib to (px,py) tuple
    projectedPtsMap = map(lambda p: (p.x, p.y), projectedPtsIter.parts())
    projectedPtsList = list(projectedPtsMap)

    return projectedPtsList, bb