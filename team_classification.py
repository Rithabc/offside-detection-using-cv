import json
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def load_team_classifications(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_image_poses(data, image_id):
    for item in data:
        if item['Image_ID'] == image_id:
            return item['Pose']
    return None

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

def reclassify_poses_with_dbscan(img, poses):
    all_players = []
    for team_name in ['Team1', 'Team2', 'GK']:
        if team_name in poses:
            for p in poses[team_name]:
                all_players.append(p.get('geometry', []))
                
    colors = []
    valid_players = []
    
    for pts in all_players:
        if not pts: continue
        color = get_dominant_color(img, pts)
        if color is not None:
            colors.append(color)
            valid_players.append(pts)
            
    if not colors:
        return {'Team1': [], 'Team2': [], 'GK': []}
        
    colors = np.array(colors)
    db = DBSCAN(eps=60, min_samples=2).fit(colors)
    labels = db.labels_
    
    label_counts = {}
    for label in labels:
        if label != -1:
            label_counts[label] = label_counts.get(label, 0) + 1
            
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    team1_label = sorted_labels[0][0] if len(sorted_labels) > 0 else -1
    team2_label = sorted_labels[1][0] if len(sorted_labels) > 1 else -1
    
    new_poses = {'Team1': [], 'Team2': [], 'GK': []}
    
    for i, label in enumerate(labels):
        player_dict = {'geometry': valid_players[i]}
        if label == team1_label:
            new_poses['Team1'].append(player_dict)
        elif label == team2_label:
            new_poses['Team2'].append(player_dict)
        else:
            new_poses['GK'].append(player_dict)
            
    return new_poses
