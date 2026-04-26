from vanishing_point import compute_vertical_vanishing_point
import cv2

img = cv2.imread('Offside_Images/1.jpg')
vp = compute_vertical_vanishing_point(img, 'right')
print(f"Vanishing point for 1.jpg: {vp}")
