import json
with open('c:/Users/psayo/Downloads/Offside_detection_dataset/final_data.json', 'r') as f:
    data = json.load(f)
print(json.dumps(data[0], indent=2)[:500])
