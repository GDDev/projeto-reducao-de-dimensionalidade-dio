import cv2
import numpy as np

imgOrigin = cv2.imread('pretzel93.png')
cv2.imshow("Imagem original", imgOrigin)
cv2.waitKey(0)

imgGray = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem cinza", imgGray)
cv2.waitKey(0)

imgBinary = cv2.GaussianBlur(imgGray, (7, 7), 0)
(T, bin) = cv2.threshold(imgBinary, 160, 255, cv2.THRESH_BINARY)
(T, binI) = cv2.threshold(imgBinary, 160, 255, cv2.THRESH_BINARY_INV)
imgBinary = np.vstack([
    np.hstack([imgBinary, bin]),
    np.hstack([binI, cv2.bitwise_and(imgBinary, imgBinary, mask = binI)])
])
cv2.imshow("Imagem binaria", imgBinary)
cv2.waitKey(0)
