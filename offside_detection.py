def get_offside_decision(pose_estimations, attackingTeamId, defendingTeamId, isKeeperFound):
    """
    Novel computation algorithm to find all players in an offside position.
    pose_estimations: list of [id, teamId, keyPoints, leftmostPoint, angleAtVanishingPoint]
    """
    # 1. Get the last defending man
    currMinAngle = 360.0
    last_defending_man = -1
    
    for pose in pose_estimations:
        if pose[1] in [defendingTeamId, 2]: # 2 is GK
            # The angle is the last element (pose[-1])
            if pose[-1] < currMinAngle:
                currMinAngle = pose[-1]
                last_defending_man = pose[0]
                
    # Exclude the last man (who is typically the goalkeeper or furthest defender) 
    # to find the second last man for the actual offside line
    exclude_last_man_id = last_defending_man
    currMinAngle = 360.0
    last_defending_man = -1
    
    for pose in pose_estimations:
        if (pose[1] == defendingTeamId or pose[1] == 2) and pose[0] != exclude_last_man_id:
            if pose[-1] < currMinAngle:
                currMinAngle = pose[-1]
                last_defending_man = pose[0]
                
    # 2. Get decision for each detected player
    for pose in pose_estimations:
        # Check attacking team
        if pose[1] == attackingTeamId:
            # If the attacker's angle is smaller than the second last defender's angle, they are offside
            if pose[-1] < currMinAngle:
                pose.append('off')
            else:
                pose.append('on')
        # Check defending team
        else:
            if pose[1] == 3: # Referee
                pose.append('ref')
            else:
                pose.append('def')

    return pose_estimations, last_defending_man

def detect_offside_players(pose_estimations):
    """
    Evaluates all pose_estimations and identifies:
      1) The last defending man
      2) Which attacking players are offside
      
    pose_estimations format: [p_id, teamId, pts, leftmost, angle]
    returns: updated pose_estimations with decision appended, and the p_id of the last defending man
    
    The dataset consistently annotates Team1 (id=0) as the attacking team
    and Team2 (id=1) as the defending team.
    """
    attackingTeamId = 0
    defendingTeamId = 1
    
    isKeeperFound = any(p[1] == 2 for p in pose_estimations)
    
    pose_estimations, last_defending_man = get_offside_decision(
        pose_estimations, 
        attackingTeamId, 
        defendingTeamId, 
        isKeeperFound
    )
    
    return pose_estimations, last_defending_man
