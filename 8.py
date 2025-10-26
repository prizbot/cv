import cv2
import numpy as np

image_path = 'star.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

output1 = image.copy()
output2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
output3 = cv2.Canny(image, 50, 150)
output4 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

corners = cv2.goodFeaturesToTrack(image, 200, 0.01, 10)
if corners is not None:
    corners = np.intp(corners)
    for x, y in corners.reshape(-1, 2):
        cv2.circle(output2, (x, y), 3, (0, 255, 0), -1)

edges_hough = cv2.Canny(image, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges_hough, 1, np.pi / 180, 150)
if lines is not None:
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(output4, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow("Original", output1)
cv2.imshow("Corners", output2)
cv2.imshow("Edges", output3)
cv2.imshow("Lines", output4)
cv2.waitKey(0)
cv2.destroyAllWindows()
