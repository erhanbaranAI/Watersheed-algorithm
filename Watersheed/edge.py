import numpy as np
import cv2
import imutils

img = cv2.imread('data/coins.jpg')

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("Output", img)

shifted = cv2.pyrMeanShiftFiltering(img, 21, 71)
cv2.imshow("shifted", shifted) #https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#pyrmeanshiftfiltering


edges1 = cv2.Canny(image=shifted,threshold1=127,threshold2=127)
cv2.imshow("edges with shifted", edges1)

# find contours in the thresholded image
cnts = cv2.findContours(edges1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] {} unique contours found".format(len(cnts)))

# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the contour
	((x, y), _) = cv2.minEnclosingCircle(c)
	cv2.drawContours(img, [c], -1, (255, 0, 0), 2)
# show the output image
cv2.imshow("Image contours", img)

cv2.waitKey(0)