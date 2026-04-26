import json

with open('final_data.json', 'r') as f:
    data = json.load(f)

total = len(data)
print(f'Total entries in JSON: {total}')

# Check if images 0-49 are all named numerically or if there's a pattern
# The original paper likely uses even=offside, odd=onside or similar
# Let's check the original repo for ground truth
