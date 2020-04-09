import numpy as np
import cv2
from scipy.optimize import fsolve


def distortImage(img, height, width, channels, xc, yc, k1, k2, k3):
    newImg = np.zeros(
        (round(height * 1.5), round(width * 1.5), channels), dtype=np.uint8
    )

    for row in range(0, height):
        for col in range(0, width):
            r = ((col - xc) ** 2 + (row - yc) ** 2) ** 0.5
            xdis = round((col - xc) * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6))
            ydis = round((row - yc) * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6))
            # print(xdis,ydis)
            newImg[
                ydis + round(height * 1.5) // 2, xdis + round(width * 1.5) // 2
            ] = img[row, col]
            # newImg[ydis, xdis] = img[row, col]

    return newImg


def undistortImage(ver, img, height, width, channels, xc, yc, k1, k2, k3):
    def equations(p):
        x, y = p
        return (
            (x - xc)
            * (
                1
                + k1 * (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5) ** 2
                + k2 * (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5) ** 4
                + k3 * (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5) ** 6
            )
            - xdis,
            (y - yc)
            * (
                1
                + k1 * (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5) ** 2
                + k2 * (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5) ** 4
                + k3 * (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5) ** 6
            )
            - ydis,
        )

    newImg = np.zeros(
        (round(height * 1.25), round(width * 1.25), channels), dtype=np.uint8
    )

    for ydis in range(0, round(height * 1.5)):
        for xdis in range(0, round(width * 1.5)):
            xdis = xdis - round(width * 1.5) // 2
            ydis = ydis - round(height * 1.5) // 2

            x, y = fsolve(equations, (1, -1))

            x = int(round(x))
            y = int(round(y))

            xdis = xdis + round(width * 1.5) // 2
            ydis = ydis + round(height * 1.5) // 2

            newImg[y, x] = img[ydis, xdis]

    return newImg


def main():
    orig = cv2.imread("picture.jpg", -1)

    height, width, channels = orig.shape

    yc = height // 2
    xc = width // 2

    k1 = -1 * 10 ** (-9)
    k2 = 1 * 10 ** (-14)
    k3 = 1 * 10 ** (-16)

    distort = distortImage(orig, height, width, channels, xc, yc, k1, k2, k3)

    # undistort_k1 = undistortImage(0, orig, height, width, channels, xc, yc, k1, k2, k3)
    # undistort_k1k2 = undistortImage(1, orig, height, width, channels, xc, yc, k1, k2, k3)
    undistort_k1k2k3 = undistortImage(
        2, distort, height, width, channels, xc, yc, k1, k2, k3
    )

    cv2.imshow("original", orig)
    cv2.imshow("distorted", distort)
    cv2.imshow("undistorted", undistort_k1k2k3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True


main()
