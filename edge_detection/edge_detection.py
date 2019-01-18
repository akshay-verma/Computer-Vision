import os
import argparse
import cv2
import numpy as np


def showImage(image, name="default"):
    cv2.destroyWindow(name)
    cv2.imshow(name, image)
    cv2.waitKey(0)


def convolution(matX, matY):
    result = 0
    totalRow, totalCol = matX.shape
    rows, cols = matY.shape
    newKernel = np.zeros((rows, cols))
    indexer = dict([i for i in zip(range(rows), reversed(range(rows)))])
    for row in range(rows):
        for col in range(cols):
            newKernel[row, col] = matY[indexer[row], indexer[col]]
    for row in range(totalRow):
        for col in range(totalCol):
            result += matX[row][col] * matY[row][col]
    return result


def computeSobelFilter(img, sobelOperator):
    totalRow, totalCol = img.shape
    # Padding
    paddedImg = np.zeros(shape=(totalRow + 2, totalCol + 2))
    paddedImg[1:totalRow + 1, 1:totalCol + 1] = img
    newImg = np.ndarray((totalRow, totalCol))
    for ir, row in enumerate(range(totalRow)):
        for ic, col in enumerate(range(totalCol)):
            cropImg = paddedImg[row:row + 3, col:col + 3]
            newImg[ir, ic] = convolution(cropImg, sobelOperator)
    return newImg


def findMaximum(img):
    rows, cols = img.shape
    tmp = None
    for row in range(rows):
        for col in range(cols):
            if tmp is None:
                tmp = abs(img[row, col])
            else:
                if abs(img[row, col]) > tmp:
                    tmp = abs(img[row, col])
    return tmp


def normalizeImage(img):
    maxVal = findMaximum(img)
    for row in range(rows):
        for col in range(cols):
            img[row, col] = abs(img[row, col]) / maxVal
    return img


def detectVerticalEdges(image):
    """
    Detects Vertical edges in the image using sobel operator

    Args:
        image(str): Location of image on the host
    """
    print("Detecting vertical edges using sobel filter...")
    sobelXOperator = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img = normalizeImage(computeSobelFilter(image, sobelXOperator))
    rows, cols = img.shape
    print("Size of image showing vertical edges: {}x{}".format(rows, cols))
    name = "image_vertical_edges.png"
    print("Saving image as {}".format(name))
    cv2.imwrite(name, img * 255)


def detectHorizontalEdges(image):
    """
    Detects Horizontal edges in the image using sobel operator

    Args:
        image(str): Location of image on the host
    """
    print("Detecting horizontal edges using sobel filter...")
    sobelYOperator = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img = normalizeImage(computeSobelFilter(image, sobelYOperator))
    rows, cols = img.shape
    print("Size of image showing Horizontal edges: {}x{}".format(rows, cols))
    name = "image_horizontal_edges.png"
    print("Saving image as {}".format(name))
    cv2.imwrite(name, img * 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image",
                        default="image.png",
                        help="Location of image on host")
    args, __ = parser.parse_known_args()
    if os.path.exists(args.image):
        print("Reading image: {}".format(args.image))
        img = cv2.imread(args.image, 0)
        rows, cols = img.shape
        print("Size of original image: {}x{}".format(rows, cols))
        detectVerticalEdges(img)
        print("=" * 80)
        detectHorizontalEdges(img)
    else:
        raise AssertionError("Input image: {} does not exists!".format(args.image))
