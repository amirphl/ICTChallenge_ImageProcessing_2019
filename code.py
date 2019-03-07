import cv2

im = cv2.imread('sample_images/39.jpg')
im = cv2.resize(im, (500, 300))
cv2.imwrite("g.jpg", im)
