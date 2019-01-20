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


import argparse
import os
import cv2
import random
import numpy as np


def performColorQuanitization(img, clusterCenter, K):
    """
    Main method which performs color quantization using K-means

    Args:
        img(str): Location of image on which color quantization is to be performed
        clusterCenter(list): Initial cluster centers (randomly selected)
        K(int): Number of clusters

    Returns:
        clusterCenter(list): Final cluster centers
        classificationVector(list): Indicates to which cluster a pixel belongs to
    """
    error = 1
    while(error > 0):
        classificationVector = classifyImagePoints(img, clusterCenter)
        newClusterCenter = updateColor(img, classificationVector, K)
        error = np.linalg.norm(newClusterCenter - clusterCenter, axis=None)
        clusterCenter = newClusterCenter
    return clusterCenter, classificationVector


def classifyImagePoints(img, clusterCenter):
    """
    Computes distance of each pixel color from the cluster center and assigns pixel
    to the closest cluster

    Args:
        img(str): Location of image on which color quantization is to be performed
        clusterCenter(list): Initial cluster centers (randomly selected)

    Returns:
        classificationVector(list): Indicates to which cluster a pixel belongs to
    """
    classificationVector = {}
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            pixel = img[row, col]
            distance = np.linalg.norm(np.uint8([pixel]) - clusterCenter, axis=1)
            classificationVector[(row, col)] = np.argmin(distance)
    return classificationVector


def updateColor(img, classificationVector, K):
    """
    Updates cluster centers by taking mean of the color of the pixel that belongs to the cluster

    Args:
        img(str): Location of image on which color quantization is to be performed
        classificationVector(list): Indicates to which cluster a pixel belongs to
        K(int): Number of clusters

    Returns:
        clusterCenter(list): Updated center of clusters
    """
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
    for K in noOfCluster:
        K = int(K)
        print("Running image quantization for K={}".format(
              K))
        img = cv2.imread(args.image, 1)
        clusterCenter = []
        for i in range(K):
            x, y = random.sample(range(0, img.shape[0]), 2)
            clusterCenter.append(img[x, y])
        clusterCenter, classificationVector = performColorQuanitization(
            img, clusterCenter, K)
        print(clusterCenter)
        rows, cols = img.shape[:2]
        for row in range(rows):
            for col in range(cols):
                img[row, col] = clusterCenter[classificationVector[row, col]]
        imageBaseName = os.path.basename(args.image)
        filename, fileExt = os.path.splitext(imageBaseName)
        filename = "{}_{}.png".format(filename, K)
        cv2.imwrite(filename, img)
        print("Image size after quantization: {}".format(os.path.getsize(filename)))
