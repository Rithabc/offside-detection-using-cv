import json, cv2, numpy as np
from team_classification import load_team_classifications, get_image_poses
from vanishing_point import compute_vertical_vanishing_point
from farthest_projection import process_team_farthest_points

data = load_team_classifications('final_data.json')

# Debug: check angle distribution for missed images
missed = ['1.jpg','3.jpg','4.jpg','5.jpg','7.jpg','8.jpg']
for img_id in missed[:3]:
    img = cv2.imread(f'Offside_Images/{img_id}')
    poses = get_image_poses(data, img_id)
    if not poses: continue
    goalDirection = 'right'
    gk = poses.get('GK', [])
    if gk and len(gk) > 0:
        gk_x = gk[0]['geometry'][0]['x']
        if gk_x < img.shape[1]/2: goalDirection = 'left'
    vp = compute_vertical_vanishing_point(img, goalDirection)
    pe = process_team_farthest_points(poses, vp, goalDirection)
    
    print(f'\n=== {img_id} (goal={goalDirection}, VP={vp[0]:.0f},{vp[1]:.0f}) ===')
    for p in pe:
        pid, tid, _, lm, ang = p
        team_name = ['T1','T2','GK'][tid]
        print(f'  {team_name} p{pid}: x={lm[1]:.0f} angle={ang:.2f}')
