import cv2
import numpy as np

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return [x, y]

def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            intersection = line_intersection(line_1, line_2)
            if intersection:
                intersections.append(intersection)
    return intersections

def get_vertical_lines(img, side):
    selectedLines = []
    selectedLinesParams = []
    linesFound = False
    BlueRedMask = 100
    
    # Adaptive thresholding for green field mask
    while not linesFound and BlueRedMask > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, BlueRedMask, BlueRedMask), (70, 255, 255))
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        
        edges = cv2.Canny(green, 150, 250, apertureSize=3) 
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None and len(lines) > 2:  
            linesFound = True  
        else: 
            BlueRedMask -= 10

    if not linesFound:
        return []

    linesFound = False
    if side == 'left':
        # Lines leaning right (goal on left)
        angleMinLimit = 20
        angleMaxLimit = 70
    else:
        # Lines leaning left (goal on right)
        angleMinLimit = 105
        angleMaxLimit = 150
    
    rLimit = 300
    while not linesFound:
        for line in lines: 
            for r, theta in line:
                isLineValid = True
                a = np.cos(theta) 
                b = np.sin(theta)
                
                # angle in degrees
                angle_deg = theta * 180 / np.pi 
                
                if angleMinLimit < angle_deg < angleMaxLimit:
                    x0 = a*r 
                    y0 = b*r 
                    x1 = int(x0 + 1000*(-b)) 
                    y1 = int(y0 + 1000*(a)) 
                    x2 = int(x0 - 1000*(-b)) 
                    y2 = int(y0 - 1000*(a))
                    
                    if len(selectedLines) > 0: 
                        for lineParams in selectedLinesParams:
                            if abs(lineParams[0] - r) < rLimit:
                                isLineValid = False
                        for selectedLine in selectedLines:
                            if not line_intersection(selectedLine, [[x1, y1], [x2, y2]]):
                                isLineValid = False
                                
                    if isLineValid:
                        selectedLines.append([[x1, y1], [x2, y2]])
                        selectedLinesParams.append([r, theta])
                        
        if len(selectedLines) < 2:
            if rLimit >= 75:
                rLimit -= 10
            else:
                angleMinLimit -= 1
                angleMaxLimit += 1
                rLimit = 100
        else:
            linesFound = True
            
    return selectedLines

def compute_vertical_vanishing_point(img, goalDirection='right'):
    """
    Compute vertical vanishing lines and vanishing point using the markings on the field.
    Returns the vanishing point (x, y) by extracting pitch markings (Hough lines)
    and calculating their intersection convergence.
    """
    selectedLines = get_vertical_lines(img, goalDirection)
    if not selectedLines:
        # Fallback if no lines are found
        return (img.shape[1]/2, -1000)
        
    intersectionPoints = find_intersections(selectedLines)
    
    if not intersectionPoints:
        return (img.shape[1]/2, -1000)

    vanishingPointX = 0.0
    vanishingPointY = 0.0
    for point in intersectionPoints:
        vanishingPointX += point[0]
        vanishingPointY += point[1]

    return (vanishingPointX / len(intersectionPoints), vanishingPointY / len(intersectionPoints))
