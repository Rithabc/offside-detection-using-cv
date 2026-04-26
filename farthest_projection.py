import numpy as np

def get_angle(vanishing_point, test_point, goalDirection='right'):
    """
    Calculate the angle between the horizontal line passing through the vanishing point 
    and the line connecting the vanishing point to the test point.
    """
    reference_point = (0.0, vanishing_point[1])
    a = np.array(reference_point)
    b = np.array(vanishing_point)
    c = np.array(test_point)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    
    if goalDirection == 'left':
        if reference_point[0] > vanishing_point[0]:
            angle = -1 * angle
    elif goalDirection == 'right':
        if reference_point[0] < vanishing_point[0]:
            angle = -1 * angle     
            
    return angle

def get_farthest_point(polygon_pts, vp, goalDirection='right'):
    """
    Find the farthest horizontal projection of the playable body parts of a player
    relative to the goal direction and vanishing point.
    """
    best_pt = None
    best_angle = 360.0
    for pt in polygon_pts:
        # Calculate angle of the line between vp and pt
        angle = get_angle(vp, (pt['x'], pt['y']), goalDirection)
        if angle < best_angle:
            best_angle = angle
            best_pt = pt
    return best_pt, best_angle

def process_team_farthest_points(poses, vp, goalDirection='right'):
    """
    Given poses clustered by team, find farthest points for all players.
    Returns format required for offside decision:
    [id, teamId, keyPoints, leftmostPoint, angleAtVanishingPoint]
    """
    pose_estimations = []
    player_id = 0
    # Map teams to IDs for the offside detection logic
    team_mapping = {'Team1': 0, 'Team2': 1, 'GK': 2}
    
    for team, teamId in team_mapping.items():
        if team not in poses:
            continue
        for player in poses[team]:
            pts = player['geometry']
            farthest_pt, angle = get_farthest_point(pts, vp, goalDirection)
            if farthest_pt:
                # leftmostPoint format [y, x]
                pose_estimations.append([player_id, teamId, pts, [farthest_pt['y'], farthest_pt['x']], angle])
                player_id += 1
                
    return pose_estimations
