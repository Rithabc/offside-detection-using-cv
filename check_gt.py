import json
with open('final_data.json', 'r') as f:
    data = json.load(f)

# Check what keys exist beyond Pose
print('Keys in first item:', list(data[0].keys()))

# Check if there is any ground truth field
for key in data[0].keys():
    if key not in ['Image_ID', 'Pose']:
        print(f'{key}: {data[0][key]}')

# Check image names for offside/onside clues
img_names = [d['Image_ID'] for d in data[:50]]
print('\nImage names:', img_names)
