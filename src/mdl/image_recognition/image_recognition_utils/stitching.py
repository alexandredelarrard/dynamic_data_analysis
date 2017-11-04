# -*- coding: utf-8 -*-

import numpy as np
import imutils
import cv2
from time import time
import matplotlib.pyplot as plt
from scipy.misc import toimage
from PIL import Image
import logging

class Stitcher:
    def __init__(self, images):
        # determine if we are using OpenCV v3.X
        self.isv3 = False
        self.images= images
        self.showMatches = False
        self.result = self.stitch()

    def stitch(self, ratio=0.75, reprojThresh=4.0):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
         imageA = np.array(self.images[0])
         imageB = np.array(self.images[1])

         imageA_re = cv2.resize(imageA, (800, 500), interpolation = cv2.INTER_CUBIC)
         imageB_re = cv2.resize(imageB, (800, 500), interpolation = cv2.INTER_CUBIC)

         (kpsA, featuresA) = self.detectAndDescribe(imageA_re)
         (kpsB, featuresB) = self.detectAndDescribe(imageB_re)

         # match features between the two images
         M, decalage_B = self.matchKeypoints(kpsA, kpsB,featuresA, featuresB, ratio, reprojThresh)

         # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
         if M is None:
            return [imageA, imageB]

         else:
            (matches, H, status) = M

            if self.showMatches:
                plt.figure(figsize=(15,15))
                vis = self.drawMatches(imageA_re, imageB_re, kpsA, kpsB, matches,status)
                plt.imshow(vis)
                plt.show()

            if len(matches) >100:
                return [imageA , imageB[:,int(decalage_B*imageB.shape[1]/600):,:]]

            else:
                return [imageA, imageB]

            #result = cv2.warpPerspective(imageA_re, H, (imageA_re.shape[1] + imageB_re.shape[1], imageA_re.shape[0]))
            #result[0:imageB_re.shape[0], 0:imageB_re.shape[1]] = imageB_re

        #return (result, vis, [imageB , imageA[:,int(400-H[0][2])*imageA.shape[1]/400:,:]])#

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
        # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SURF")
            kps = detector.detect(gray)

        # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SURF")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
            kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
            return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 40 matches
        if len(matches) > 40:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status), np.percentile(zip(*ptsB), 30)

        # otherwise, no homograpy could be computed
        return None, 0

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis


def check_stitching_ref(ref_segmentation):

    keys_suppressed = []

    for i in range(len(ref_segmentation)-1):
        
        logging.error(i)
        stitcher = Stitcher([ref_segmentation.values()[i],
                             ref_segmentation.values()[i+1]])
        images= stitcher.result

        ### second is included in first
        if images[1].shape[1]<50:
            keys_suppressed.append(ref_segmentation.keys()[i+1])

        else:
            ref_segmentation[ref_segmentation.keys()[i]] = toimage(images[0])
            ref_segmentation[ref_segmentation.keys()[i+1]] = toimage(images[1])

    for i in keys_suppressed:
        logging.error("picture %s has been stitched"%i)
        ref_segmentation.pop(i)

    return ref_segmentation