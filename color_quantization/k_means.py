"""
MIT License
Copyright (c) 2019 Akshay Verma
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import cv2
import argparse
import random
import numpy as np


def performColorQuanitization(img, clusterCenter, noOfCluster):
    error = 1
    while(error > 0):
        classificationVector = classifyImagePoints(img, clusterCenter)
        newClusterCenter = updateColor(img, classificationVector, noOfCluster)
        error = np.linalg.norm(newClusterCenter - clusterCenter, axis=None)
        clusterCenter = newClusterCenter
    return clusterCenter, classificationVector


def classifyImagePoints(img, clusterCenter):
    classificationVector = {}
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            pixel = img[row, col]
            distance = np.linalg.norm(np.uint8([pixel]) - clusterCenter, axis=1)
            classificationVector[(row, col)] = np.argmin(distance)
    return classificationVector


def updateColor(img, classificationVector, K):
    clusterCenter = []
    rows, cols = img.shape[:2]
    for clusterNum in range(K):
        points = []
        for row in range(rows):
            for col in range(cols):
                if classificationVector[row, col] == clusterNum:
                    points.append(img[row, col])
        clusterCenter.append(np.round(np.mean(points, axis=0), 2))
        # clusterCenter[clusterNum] = np.round(np.mean(points, axis=0), 2)
    # return np.float32(list(clusterCenter.values()))
    return np.float32(clusterCenter)


if __name__ == "__main__":
    random.seed(2019181)
    parser = argparse.ArgumentParser()
    parser.add_argument("--noOfCluster", default="3,5,10,20", type=str,
                        help="Number of cluster to be formed")
    parser.add_argument("--image", default="baboon.jpg",
                        help="Image for color quantization")
    args, _ = parser.parse_known_args()
    noOfCluster = args.noOfCluster.split(",")
    for num in noOfCluster:
        num = int(num)
        print("Running image quantization for K={}".format(
              num))
        img = cv2.imread(args.image, 1)
        clusterCenter = []
        for i in range(num):
            x, y = random.sample(range(0, img.shape[0]), 2)
            clusterCenter.append(img[x, y])
        clusterCenter, classificationVector = performColorQuanitization(
            img, clusterCenter, num)
        print(clusterCenter)
        rows, cols = img.shape[:2]
        for row in range(rows):
            for col in range(cols):
                img[row, col] = clusterCenter[classificationVector[row, col]]
        cv2.imwrite("baboon_{}.jpg".format(num), img)
