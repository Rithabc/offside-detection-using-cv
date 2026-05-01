import os
import cv2
import numpy as np

# Import from the standalone individual modules
from team_classification import load_team_classifications, get_image_poses, get_team_dominant_color
from vanishing_point import compute_vertical_vanishing_point
from farthest_projection import process_team_farthest_points
from offside_detection import detect_offside_players

# Predefined jersey color map (BGR format)
COLOR_MAP = {
    'red':    np.array([0, 0, 255]),
    'blue':   np.array([255, 0, 0]),
    'white':  np.array([255, 255, 255]),
    'black':  np.array([30, 30, 30]),
    'yellow': np.array([0, 255, 255]),
    'green':  np.array([0, 200, 0]),
    'orange': np.array([0, 165, 255]),
    'purple': np.array([128, 0, 128]),
    'pink':   np.array([203, 192, 255]),
    'cyan':   np.array([255, 255, 0]),
    'grey':   np.array([128, 128, 128]),
    'maroon': np.array([0, 0, 128]),
}

def match_defending_team(img, poses, target_bgr):
    """
    Compare the dominant jersey color of Team1 and Team2 against the
    user-specified defending color. The closer team becomes the defender.
    Returns (attackingTeamId, defendingTeamId).
    """
    color_team1 = get_team_dominant_color(img, poses, 'Team1')
    color_team2 = get_team_dominant_color(img, poses, 'Team2')

    dist1 = np.linalg.norm(color_team1 - target_bgr) if color_team1 is not None else float('inf')
    dist2 = np.linalg.norm(color_team2 - target_bgr) if color_team2 is not None else float('inf')

    if dist1 <= dist2:
        # Team1 is the defending team
        return 1, 0   # attackingTeamId=1 (Team2), defendingTeamId=0 (Team1)
    else:
        # Team2 is the defending team
        return 0, 1   # attackingTeamId=0 (Team1), defendingTeamId=1 (Team2)

def main():
    dataset_path = '.' # Assuming running from inside Offside_detection_dataset
    json_path = os.path.join(dataset_path, 'final_data.json')
    images_dir = os.path.join(dataset_path, 'Offside_Images')
    output_dir = os.path.join(dataset_path, 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load team classifications
    data = load_team_classifications(json_path)

    # Ask user which image to process
    img_id = input("\nEnter the image filename (e.g. 0.jpg, 12.jpg): ").strip()
    
    # Find the matching entry in the dataset
    item = None
    for entry in data:
        if entry['Image_ID'] == img_id:
            item = entry
            break
    
    if item is None:
        print(f"  Image '{img_id}' not found in dataset.")
        return

    # Process the requested image
    img_path = os.path.join(images_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Could not load {img_id}.")
        return
        
    # Get classified poses
    poses = get_image_poses(data, img_id)
    if not poses:
        print(f"  No poses for {img_id}.")
        return

    # Ask for defending colour
    print(f"\n{'='*50}")
    print(f"  Image: {img_id}")
    print(f"{'='*50}")
    print(f"  Available colours: {', '.join(COLOR_MAP.keys())}")
    defending_color = input("  Enter the jersey colour of the DEFENDING team: ").strip().lower()

    while defending_color not in COLOR_MAP:
        print(f"  '{defending_color}' is not recognised.")
        defending_color = input("  Enter the jersey colour of the DEFENDING team: ").strip().lower()

    target_bgr = COLOR_MAP[defending_color]
    print(f"  → Defending: {defending_color}  (BGR {target_bgr.tolist()})")
    
    # Dynamically determine goal direction based on GK
    goalDirection = 'right'
    gk_pose = poses.get('GK', [])
    if gk_pose and len(gk_pose) > 0 and len(gk_pose[0]['geometry']) > 0:
        gk_x = gk_pose[0]['geometry'][0]['x']
        if gk_x < (img.shape[1] / 2):
            goalDirection = 'left'

    # Dynamically determine which team is defending based on jersey colour
    attackingTeamId, defendingTeamId = match_defending_team(img, poses, target_bgr)
    print(f"  → Goal direction: {goalDirection} | Attacking=Team{attackingTeamId} | Defending=Team{defendingTeamId}")
        
    # Step 2: Calculate Vertical Vanishing Point
    vp = compute_vertical_vanishing_point(img, goalDirection)
    
    # Step 3: Find farthest horizontal projection of all players
    pose_estimations = process_team_farthest_points(poses, vp, goalDirection)

    # Step 4: Find players in offside position
    pose_estimations, last_defending_man = detect_offside_players(
        pose_estimations, attackingTeamId, defendingTeamId
    )
    
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
        text = decision
        if p_id == last_defending_man:
            text = 'last man'
        elif teamId == 2:
            text = 'keep'
        
        color = (0, 0, 255) if text == 'off' else (0, 255, 0)
        cv2.putText(img, text, (leftmost[1], leftmost[0]-15), font, 1, color, 2, cv2.LINE_AA)
        cv2.putText(img, str(teamId), (leftmost[1]+20, leftmost[0]), font, 1, (200, 255, 155), 2, cv2.LINE_AA)

    out_path = os.path.join(output_dir, img_id)
    cv2.imwrite(out_path, img)
    print(f"Output image saved to {out_path}\n")

if __name__ == '__main__':
    main()
