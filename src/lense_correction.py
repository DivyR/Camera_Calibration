"""Essentially a large library written for
ESC204 - Correcting Lense Distortion Project"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


class LCImage:
    """
    Lense-Correction-Image class stores at its core a numpy array filled
    with image-pixel data.
    """

    def __init__(self, imagePath):
        # core image
        self.image = cv2.imread(imagePath)
        self.center = np.array(
            [
                math.ceil(self.image.shape[1] / 2) - 1,
                math.ceil(self.image.shape[0] / 2) - 1,
            ]
        )  # numpy shape stores (y, x) hence the flipped indices
        self.radi = {}

    def update(self, image):
        self.image = image
        self.center = np.array(
            [
                math.ceil(self.image.shape[1] / 2) - 1,
                math.ceil(self.image.shape[0] / 2) - 1,
            ]
        )  # numpy shape stores (y, x) hence the flipped indices

    def distort(self, kValues: list, name="distorted") -> np.ndarray:
        """
        Distorts the image following the vectorized formulation of
        Brown's lens-distortion model.
        kValues: list of k-values in the order of k1, k2, and k3
        Returns
        """
        # k1, k2, k3 values are stored in this numpy array
        self.kVec = np.array(kValues)
        # blank image
        blankImage, wMin, hMin = self.generate_blank_image(self.generate_distortion)
        # loop and distort
        for y_pixel in range(self.image.shape[0]):
            for x_pixel in range(self.image.shape[1]):
                distoX, distoY = self.generate_distortion(x_pixel, y_pixel)
                blankImage[distoY + hMin][distoX + wMin] = self.image[y_pixel][x_pixel]
        print(self.image.shape, blankImage.shape)
        image = self.rescale_image(blankImage)
        self.save_image(image, name)
        self.update(image)
        return image

    def generate_distortion(self, x_pixel, y_pixel):
        """
        Determines the distortion of x and y pixels using Brown's lens
        distortion model.
        """
        # normalize the coordinate
        normX = x_pixel - self.center[0]
        normY = y_pixel - self.center[1]
        # radius-sqaured
        rSquared = normX ** 2 + normY ** 2
        # apply the distortion formula
        constant = (
            1
            + self.kVec[0] * rSquared
            + self.kVec[1] * (rSquared ** 2)
            + self.kVec[2] * (rSquared ** 3)
        )
        distoX = int(round(constant * normX))
        distoY = int(round(constant * normY))
        return distoX, distoY

    def generate_blank_image(self, rule):
        """
        Generates a distortion by considering the points:
        (0 0), (0 y), (x 0), (x y)
        """
        # determine the distortion at the specified points
        whCorner = self.generate_min_max_4points(
            (0, 0),
            (0, self.image.shape[0] - 1),
            (self.image.shape[1] - 1, 0),
            (self.image.shape[1] - 1, self.image.shape[0] - 1),
            rule,
        )
        whMid = self.generate_min_max_4points(
            (0, self.center[1]),
            (self.center[0], 0),
            (self.image.shape[1] - 1, self.center[1]),
            (self.center[0], self.image.shape[0] - 1),
            rule,
        )
        # take the data with the greater width value since it accounts for the type of distortion
        whData = max(whCorner, whMid, key=lambda hw: hw[0])
        width, height, wMin, hMin = whData[0], whData[1], whData[2], whData[3]
        # return the new blank image
        return (
            # change the +2 factor to something greater if issues arise with rounding
            np.zeros(shape=[height + 2, width + 2, 3], dtype=np.int8) * 255,
            wMin,
            hMin,
        )

    def generate_min_max_4points(self, a, b, c, d, rule):
        # determine the distortion at the specified points
        dista = rule(*a)
        distb = rule(*b)
        distc = rule(*c)
        distd = rule(*d)

        # determine the height and width of the new image
        wMin = min(dista, distb, distc, distd, key=lambda point: point[0])[0]
        wMax = max(dista, distb, distc, distd, key=lambda point: point[0])[0]
        width = wMax + abs(wMin)

        hMin = min(dista, distb, distc, distd, key=lambda point: point[1])[1]
        hMax = max(dista, distb, distc, distd, key=lambda point: point[1])[1]
        height = hMax + abs(hMin)

        return (width, height, abs(wMin), abs(hMin))

    def rescale_image(self, image):
        factor = max(
            image.shape[0] / self.image.shape[0], image.shape[1] / self.image.shape[1]
        )
        newDim = (
            math.ceil(image.shape[1] / factor),
            math.ceil(image.shape[0] / factor),
        )
        # newDim = (self.image.shape[1], self.image.shape[0])
        return cv2.resize(image, newDim, interpolation=cv2.INTER_LINEAR)

    def undistort(self, kValues: list, funky=True) -> np.ndarray:
        """
        Undistorts an image with the provided k-values.
        """
        if not funky:
            # k1, k2, k3 values are stored in this numpy array
            self.kVec = np.array(kValues)
            blankImage, wMin, hMin = self.generate_blank_image(
                self.generate_undistortion
            )
            print(self.image.shape, self.center)
            print(blankImage.shape)
            for y_pix in range(self.image.shape[0]):
                for x_pix in range(self.image.shape[1]):
                    newX, newY = self.generate_undistortion(x_pix, y_pix)
                    if newX is None:
                        continue
                    blankImage[newY + hMin][newX + wMin] = self.image[y_pix][x_pix]
            image = blankImage
            self.save_image(image, "undistorted")
            self.update(image)
            return image

        kVec = [
            -kValues[0],
            3 * (kValues[0] ** 2) - kValues[1],
            -12 * (kValues[0]) ** 2 + 8 * kValues[0] * kValues[1] - kValues[1],
        ]
        return self.distort(kVec, name="undistorted")

    def generate_undistortion(self, x_pix, y_pix):
        """
        Undistorts an image using Brown's Model
        """
        normX, normY = x_pix - self.center[0], y_pix - self.center[1]
        # radius-sqaured
        rSquared = normX ** 2 + normY ** 2
        constant = (
            1
            + self.kVec[0] * rSquared
            + self.kVec[1] * (rSquared ** 2)
            + self.kVec[2] * (rSquared ** 3)
        )
        newX, newY = normX / constant, normY / constant
        return int(round(newX)), int(round(newY))

    def show(self):
        """
        Quick prototyping function to display the image.
        """
        plt.imshow(self.image)

    def save_image(self, image, name):
        cv2.imwrite(
            "exports/"
            + name
            + str(self.kVec[0])
            + "A"
            + str(self.kVec[1])
            + "A"
            + str(self.kVec[2])
            + ".png",
            image,
        )


if __name__ == "__main__":
    path = "data/img01.jpg"
    kVals = [10**-7, 10**-12, 10**-15]
    test1 = LCImage(path)
    image = test1.distort(kVals)
    print("Undistorting")
    # image = test1.undistort(kVals)
    # print("Undistorted")
    # kVals = [-kVals[0], 3 * kVals[0] ** 2, -12 * kVals[0] ** 2]
    # test2 = LCImage("exports\distorted-1.5e-06A0.0A0.0.png")
    # image = test2.undistort(kVals)
    plt.imshow(image)
    plt.show()
    pass
