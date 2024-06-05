import json
import yaml
import argparse
import chardet
from ijson import items
import regex as re


def clean_file(input_file, cleaned_file):
    with open(input_file, 'r', errors='ignore') as f:
        content = f.read()
        
    # Replace non-ASCII characters
    cleaned_content = re.sub(r'[^\x00-\x7F]+', '', content)

    cleaned_content = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_content)

    with open(cleaned_file, 'w') as f:
        f.write(cleaned_content)

def processor(input_file, output_file):

    cleaned_file = 'cleaned_file.json'
    clean_file(input_file, cleaned_file)

    processed_data = {
        'name': 'vllm_model',
        'operators': []
    }
    with open(cleaned_file, 'r', errors='replace') as json_file:
        for item in items(json_file, 'traceEvents.item'):
            try:
                if isinstance(item, dict) and "name" in item and item["name"].startswith("aten::"):
                    processed_item = {
                        "operator_name": item["name"].replace("aten::", "aten."),
                        "size": str(item["args"].get("Input Dims", [])),
                        "dtype": item["args"].get("Input type", []),
                        "dur": item.get("dur", None)
                    }
                    processed_data['operators'].append(processed_item)
            except Exception as e:
                continue

    final_data = [
        {
            'name': processed_data['name'],
            'operators': [
                {
                    'operator_name': op['operator_name'],
                    'size': op['size'],
                    'dtype': op['dtype'],
                    'dur': op['dur']
                } for op in processed_data['operators']
            ]
        }
    ]


    with open(f"{output_file}", "w") as yaml_file:
        yaml.dump(final_data, yaml_file, default_flow_style=True, sort_keys=False, indent=2, allow_unicode=True)

    print(f"Processed data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Process JSON file and save to YAML file.')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file name')
    parser.add_argument('--output', type=str, required=True, help='Output YAML file name')
    args = parser.parse_args()
    processor(args.input, args.output)




if __name__ == "__main__":
   main()

