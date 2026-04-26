"""
Precision & Recall evaluation for the offside detection pipeline.

Ground truth assumption: All images in Offside_Images/ depict an offside scenario,
meaning at least one attacking player should be classified as 'off'.
"""
import os
import cv2
import numpy as np

from team_classification import load_team_classifications, get_image_poses
from vanishing_point import compute_vertical_vanishing_point
from farthest_projection import process_team_farthest_points
from offside_detection import detect_offside_players


def run_evaluation():
    dataset_path = '.'
    json_path = os.path.join(dataset_path, 'final_data.json')
    images_dir = os.path.join(dataset_path, 'Offside_Images')

    data = load_team_classifications(json_path)

    # Ground truth: every image in the dataset IS an offside scenario
    # TP = pipeline correctly detects offside (at least 1 player flagged 'off')
    # FN = pipeline misses offside (0 players flagged 'off' when there should be)
    # FP = not applicable here since all images are offside (no non-offside images)
    #
    # We also evaluate per-image statistics for deeper analysis.

    tp = 0   # Correctly detected offside in an offside image
    fn = 0   # Failed to detect any offside in an offside image
    total_images = 0
    total_offside_players = 0
    total_onside_players = 0
    total_defenders = 0

    results = []

    for item in data[:50]:
        img_id = item['Image_ID']
        img_path = os.path.join(images_dir, img_id)
        img = cv2.imread(img_path)
        if img is None:
            continue

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

        vp = compute_vertical_vanishing_point(img, goalDirection)
        pose_estimations = process_team_farthest_points(poses, vp, goalDirection)
        pose_estimations, last_defending_man = detect_offside_players(pose_estimations)

        # Count decisions
        offside_count = sum(1 for p in pose_estimations if p[-1] == 'off')
        onside_count = sum(1 for p in pose_estimations if p[-1] == 'on')
        defender_count = sum(1 for p in pose_estimations if p[-1] == 'def')

        total_offside_players += offside_count
        total_onside_players += onside_count
        total_defenders += defender_count
        total_images += 1

        detected_offside = offside_count > 0

        if detected_offside:
            tp += 1
        else:
            fn += 1

        results.append({
            'image': img_id,
            'offside_detected': detected_offside,
            'offside_count': offside_count,
            'onside_count': onside_count,
            'defender_count': defender_count,
            'total_players': len(pose_estimations)
        })

    # Since ALL images are offside scenarios (ground truth = positive):
    # Precision = TP / (TP + FP). With no negative images, FP = 0 among true offside images.
    # But we can still measure if the system ever says "offside" when it shouldn't
    # by looking at the ratio of correct detections.
    #
    # For a meaningful metric, we compute:
    #   - Detection Rate (Recall) = TP / (TP + FN)
    #   - Since there are no non-offside images, Precision = 1.0 by definition
    #     (every detection is on a true offside image)

    precision = tp / (tp + 0) if (tp + 0) > 0 else 0  # No FP possible in this dataset
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("=" * 60)
    print("  OFFSIDE DETECTION — EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Total Images Evaluated  : {total_images}")
    print(f"  Ground Truth            : All images are offside scenarios")
    print()
    print(f"  True Positives  (TP)    : {tp}  (offside correctly detected)")
    print(f"  False Negatives (FN)    : {fn}  (offside missed)")
    print()
    print(f"  Precision               : {precision:.4f}")
    print(f"  Recall (Detection Rate) : {recall:.4f}")
    print(f"  F1 Score                : {f1:.4f}")
    print()
    print("-" * 60)
    print("  PLAYER-LEVEL STATISTICS")
    print("-" * 60)
    print(f"  Total Players Processed : {total_offside_players + total_onside_players + total_defenders}")
    print(f"  Flagged Offside         : {total_offside_players}")
    print(f"  Flagged Onside          : {total_onside_players}")
    print(f"  Flagged Defender        : {total_defenders}")
    print()

    # Per-image breakdown
    print("-" * 60)
    print("  PER-IMAGE BREAKDOWN")
    print("-" * 60)
    print(f"  {'Image':<12} {'Offside?':<10} {'#Off':<6} {'#On':<6} {'#Def':<6} {'Total':<6}")
    print(f"  {'-'*10:<12} {'-'*8:<10} {'-'*4:<6} {'-'*4:<6} {'-'*4:<6} {'-'*4:<6}")
    for r in results:
        status = "YES" if r['offside_detected'] else "NO"
        print(f"  {r['image']:<12} {status:<10} {r['offside_count']:<6} {r['onside_count']:<6} {r['defender_count']:<6} {r['total_players']:<6}")

    print("=" * 60)


if __name__ == '__main__':
    run_evaluation()
