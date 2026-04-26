import cv2
import sys
import os

# Import original repo
sys.path.append('Computer-Vision-based-Offside-Detection-in-Soccer')
from VanishingPointUtils import get_vertical_vanishing_point as original_vp
from vanishing_point import compute_vertical_vanishing_point as my_vp

img = cv2.imread('Offside_Images/0.jpg')
vp1 = original_vp(img, 'right')
vp2 = my_vp(img, 'right')

print("Original repo VP:", vp1)
print("My standalone VP:", vp2)
