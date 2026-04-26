import cv2
import numpy as np

orig = cv2.imread('Offside_Images/1.jpg')
out = cv2.imread('output/1.jpg')

# The user might have drawn something. We look for pixels that are significantly different.
diff = cv2.absdiff(orig, out)
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

# Exclude the red line we drew!
# Our red line is (0, 0, 255).
# Wait, let's just find the hough lines in the mask!
lines = cv2.HoughLines(mask, 1, np.pi / 180, 100)

if lines is not None:
    for line in lines[:5]:
        rho, theta = line[0]
        angle_deg = theta * 180 / np.pi
        print(f"Detected line angle: {angle_deg:.2f} degrees")
else:
    print("No prominent lines found in the diff.")
