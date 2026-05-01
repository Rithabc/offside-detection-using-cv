import json
import cv2
import numpy as np
import os
import sys

sys.path.append('Computer-Vision-based-Offside-Detection-in-Soccer')
from VanishingPointUtils import get_vertical_vanishing_point, get_angle
from CoreOffsideUtils import get_offside_decision

def get_farthest_point(polygon_pts, vp, image, goalDirection):
    best_pt = None
    best_angle = 360.0
    for pt in polygon_pts:
        # get_angle returns angle of the line between vp and pt relative to horizontal line
        angle = get_angle(vp, (pt['y'], pt['x']), image, goalDirection)
        if angle < best_angle:
            best_angle = angle
            best_pt = pt
    return best_pt, best_angle

def main():
    dataset_path = '../Offside_detection_dataset'
    with open(os.path.join(dataset_path, 'final_data.json'), 'r') as f:
        data = json.load(f)

    os.makedirs('output', exist_ok=True)
    goalDirection = 'right'

    for item in data[:50]:
        img_id = item['Image_ID']
        print(f"Processing {img_id}")
        img_path = os.path.join(dataset_path, 'Offside_Images', img_id)
        img = cv2.imread(img_path)
        if img is None: continue
            
        vp = get_vertical_vanishing_point(img, goalDirection)
        
        poses = item['Pose']
        
        # We will collect all players with their "farthest pt" angle
        pose_estimations = [] # format for get_offside_decision: [id, teamId, keyPoints, leftmostPoint, angleAtVanishingPoint]
        
        player_id = 0
        for team, teamId in [('Team1', 0), ('Team2', 1), ('GK', 2)]:
            if team not in poses: continue
            for player in poses[team]:
                pts = player['geometry']
                farthest_pt, angle = get_farthest_point(pts, vp, img, goalDirection)
                if farthest_pt:
                    pose_estimations.append([player_id, teamId, pts, [farthest_pt['y'], farthest_pt['x']], angle])
                    player_id += 1

        # We assume Team1 is attacking (0), Team2 is defending (1). The repository uses 0 for attacking and 1 for defending. 
        
        isKeeperFound = any(p[1] == 2 for p in pose_estimations)
        
        pose_estimations, last_defending_man = get_offside_decision(pose_estimations, vp, 0, 1, isKeeperFound)
        
        # Draw everything
        cv2.circle(img, (int(vp[0]), int(vp[1])), 10, (0, 255, 0), -1) #"vanishing point(green)"
        
        for pose in pose_estimations:
            p_id, teamId, pts, leftmost, angle, decision = pose
            
            # polygon
            poly = np.array([[pt['x'], pt['y']] for pt in pts], np.int32)
            cv2.polylines(img, [poly], True, (255, 255, 255), 1)
            
            # farthest pt
            cv2.circle(img, (leftmost[1], leftmost[0]), 5, (255, 0, 0), -1) # Blue farthest pt
            
            # line from vp
            cv2.line(img, (int(vp[0]), int(vp[1])), (leftmost[1], leftmost[0]), (0, 0, 255), 2) # Red line from vp
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            # write decision
            text = ''
            if teamId == 0: # attacker
                text = 'off' if decision == 'off' else 'on'
            elif teamId == 1: # defender
                text = 'last man' if p_id == last_defending_man else 'def'
            elif teamId == 2:
                text = 'keep'
            
            color = (0, 0, 255) if text == 'off' else (0, 255, 0)
            cv2.putText(img, text, (leftmost[1], leftmost[0]-15), font, 1, color, 2, cv2.LINE_AA)
            cv2.putText(img, str(teamId), (leftmost[1]+20, leftmost[0]), font, 1, (200, 255, 155), 2, cv2.LINE_AA)

        cv2.imwrite(f"output/{img_id}", img)

if __name__ == '__main__':
    main()
