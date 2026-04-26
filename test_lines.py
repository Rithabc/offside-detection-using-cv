import cv2
import numpy as np

img = cv2.imread('Offside_Images/1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
edges = cv2.Canny(mask, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

angles = []
if lines is not None:
    for line in lines:
        angles.append(line[0][1] * 180 / np.pi)

print(f"Prominent line angles: {angles[:20]}")
