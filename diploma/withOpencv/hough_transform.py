import cv2
import numpy as np

img = cv2.imread('single_sphere/pic20.png', 0)
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
final_img = cv2.imread('single_sphere/pic20.png')
for i in circles[0, :]:
    # draw the outer circle
    print(i[0])
    print(i[1])
    print(i[2])
    cv2.circle(final_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(final_img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
