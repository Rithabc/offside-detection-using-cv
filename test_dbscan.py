import json
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def get_dominant_color(img, pts):
    pts_array = np.array([[pt['x'], pt['y']] for pt in pts], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts_array)
    roi = img[y:y+h, x:x+w]
    if roi.size == 0: return None
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv_roi, lower_green, upper_green)
    player_mask = cv2.bitwise_not(mask)
    player_pixels = roi[player_mask > 0]
    
    if len(player_pixels) == 0: return None
    return np.mean(player_pixels, axis=0)

with open('c:/Users/psayo/Downloads/Offside_detection_dataset/final_data.json', 'r') as f:
    data = json.load(f)

item = data[1]
img = cv2.imread('c:/Users/psayo/Downloads/Offside_detection_dataset/Offside_Images/' + item['Image_ID'])
poses = item['Pose']

all_players = []
for team_name in ['Team1', 'Team2', 'GK']:
    if team_name in poses:
        for p in poses[team_name]:
            all_players.append(p.get('geometry', []))

colors = []
for pts in all_players:
    c = get_dominant_color(img, pts)
    if c is not None: colors.append(c)

if colors:
    colors = np.array(colors)
    print("Colors extracted:")
    for c in colors: print(c)
    
    for eps in [20, 30, 40, 50, 60, 80]:
        db = DBSCAN(eps=eps, min_samples=2).fit(colors)
        print(f"eps={eps}, labels={db.labels_}")
