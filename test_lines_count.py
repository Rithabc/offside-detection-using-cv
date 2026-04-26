from vanishing_point import get_vertical_lines
import cv2
import numpy as np

img = cv2.imread('Offside_Images/1.jpg')
lines = get_vertical_lines(img, 'right')
for line in lines:
    for l in line:
        theta = l[1]
        print(theta * 180 / np.pi)
