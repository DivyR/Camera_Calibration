import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread("exports\dist11e-09Z1e-14Z1e-16.png")
plt.imshow(img)
plt.show()
# bilinear interperlation to fix 'holes' in the image
def bilinear_inter(image, tol=0):
    height, width, _ = image.shape
    for pX in range(width):
        for pY in range(height):
            if not np.allclose(image[pY][pX], [0, 0, 0], atol=tol):
                continue
            try:
                image[pY][pX] = (
                    (image[pY - 1][pX]) // 4
                    + (image[pY + 1][pX]) // 4
                    + (image[pY][pX - 1]) // 4
                    + (image[pY][pX + 1]) // 4
                    # + (image[pY - 1][pX - 1]) // 8
                    # + (image[pY - 1][pX + 1]) // 8
                    # + (image[pY + 1][pX - 1]) // 8
                    # + (image[pY + 1][pX + 1]) // 8
                )
            except:
                continue
    return image


for i in range(1):
    img = bilinear_inter(img, i)
    cv2.imwrite("exports/test/bi" + str(1) + ".png", img)
    print("Done", i)
