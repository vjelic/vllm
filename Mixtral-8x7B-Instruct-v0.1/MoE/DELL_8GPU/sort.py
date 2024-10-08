import json
import sys

# check the input path
if len(sys.argv) != 2:
    print("Usage: python sort.py filename")
    sys.exit(1)

# Load the JSON data from a file
filename = sys.argv[1]
with open(filename, 'r') as file:
    data = json.load(file)

# Sort the dictionary by keys
sorted_data = dict(sorted(data.items(), key=lambda item: int(item[0])))

# # Save the sorted dictionary to a new JSON file
with open(filename, 'w') as file:
    json.dump(sorted_data, file, indent=4)

# print("JSON data has been sorted and saved to 'sorted_output.json'.")
