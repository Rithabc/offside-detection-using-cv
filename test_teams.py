import json
import numpy as np

with open('c:/Users/psayo/Downloads/Offside_detection_dataset/final_data.json', 'r') as f:
    data = json.load(f)

for i in range(10):
    item = data[i]
    print(f"Image {item['Image_ID']}")
    pose = item['Pose']
    gk_pts = pose.get('GK', [])
    if not gk_pts:
        print('No GK')
        continue
    gk_x = np.mean([pt['x'] for pt in gk_pts[0]['geometry']])
    
    t1_min_dist = min([abs(np.mean([pt['x'] for pt in p['geometry']]) - gk_x) for p in pose.get('Team1', [])]) if pose.get('Team1') else 9999
    t2_min_dist = min([abs(np.mean([pt['x'] for pt in p['geometry']]) - gk_x) for p in pose.get('Team2', [])]) if pose.get('Team2') else 9999
    
    print(f"Team1 min dist: {t1_min_dist:.1f}, Team2 min dist: {t2_min_dist:.1f}")
    print(f"Defending Team is: {'Team1' if t1_min_dist < t2_min_dist else 'Team2'}")
