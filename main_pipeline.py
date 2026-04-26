import os
import cv2
import numpy as np

# Import from the standalone individual modules
from team_classification import load_team_classifications, get_image_poses
from vanishing_point import compute_vertical_vanishing_point
from farthest_projection import process_team_farthest_points
from offside_detection import detect_offside_players

def main():
    dataset_path = '.' # Assuming running from inside Offside_detection_dataset
    json_path = os.path.join(dataset_path, 'final_data.json')
    images_dir = os.path.join(dataset_path, 'Offside_Images')
    output_dir = os.path.join(dataset_path, 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load team classifications
    data = load_team_classifications(json_path)

    for item in data[:50]:
        img_id = item['Image_ID']
        print(f"Processing {img_id}")
        img_path = os.path.join(images_dir, img_id)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Get classified poses
        poses = get_image_poses(data, img_id)
        if not poses:
            continue
        
        # Dynamically determine goal direction based on GK
        goalDirection = 'right'
        gk_pose = poses.get('GK', [])
        if gk_pose and len(gk_pose) > 0 and len(gk_pose[0]['geometry']) > 0:
            gk_x = gk_pose[0]['geometry'][0]['x']
            if gk_x < (img.shape[1] / 2):
                goalDirection = 'left'
            
        # Step 2: Calculate Vertical Vanishing Point
        vp = compute_vertical_vanishing_point(img, goalDirection)
        
        # Step 3: Find farthest horizontal projection of all players
        pose_estimations = process_team_farthest_points(poses, vp, goalDirection)

        # Step 4: Find players in offside position
        pose_estimations, last_defending_man = detect_offside_players(pose_estimations)
        
        # Step 5: Visualize
        cv2.circle(img, (int(vp[0]), int(vp[1])), 10, (0, 255, 0), -1) 
        
        for pose in pose_estimations:
            p_id, teamId, pts, leftmost, angle, decision = pose
            
            # Polygon
            poly = np.array([[pt['x'], pt['y']] for pt in pts], np.int32)
            cv2.polylines(img, [poly], True, (255, 255, 255), 1)
            
            # Farthest point (blue)
            cv2.circle(img, (leftmost[1], leftmost[0]), 5, (255, 0, 0), -1)
            
            # Only draw offside lines for the last man and offside players
            if decision == 'off' or p_id == last_defending_man:
                vp_x, vp_y = int(vp[0]), int(vp[1])
                pt_x, pt_y = leftmost[1], leftmost[0]
                
                line_color = (0, 255, 255) if p_id == last_defending_man else (0, 0, 255)
                
                if vp_x != pt_x:
                    slope = (pt_y - vp_y) / (pt_x - vp_x)
                    intercept = vp_y - slope * vp_x
                    
                    height, width = img.shape[:2]
                    y_at_x0 = int(intercept)
                    y_at_xw = int(slope * width + intercept)
                    
                    cv2.line(img, (0, y_at_x0), (width, y_at_xw), line_color, 2)
                else:
                    cv2.line(img, (pt_x, 0), (pt_x, img.shape[0]), line_color, 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            # write decision based on the dynamically assigned decision string
            text = decision
            if p_id == last_defending_man:
                text = 'last man'
            elif teamId == 2:
                text = 'keep'
            
            color = (0, 0, 255) if text == 'off' else (0, 255, 0)
            cv2.putText(img, text, (leftmost[1], leftmost[0]-15), font, 1, color, 2, cv2.LINE_AA)
            cv2.putText(img, str(teamId), (leftmost[1]+20, leftmost[0]), font, 1, (200, 255, 155), 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(output_dir, img_id), img)

if __name__ == '__main__':
    main()
