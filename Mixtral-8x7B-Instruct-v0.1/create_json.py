import json
import argparse

# Create a dictionary
data_dict = {
    "Machine": "",
}

def CreateJson(args):
    # Create throughput json
    print(f"Create benchmark.json for customer {args.customer}")

    if args.customer == "SNOW" or args.customer == "ServiceNow":
        input_enum = [2500, 4400, 6000, 11000]
        output_enum = [300, 500]
        batch_enum = ["BS_" + str(bs) for bs in [1, 4, 8, 16, 24, 32, 40, 48, 56, 64]]
        TP_type = ["TP_" + str(tp) for tp in [8]]

        for TP in TP_type:
            data_dict[TP] = dict()
            for bs in batch_enum:
                data_dict[TP][bs] = dict()

        # Fill in json
        for input_len in input_enum:
            for output_len in output_enum:
                test_name = "input_len_" + str(input_len) + "_output_len_" + str(output_len)
                for TP in TP_type:
                    for bs in batch_enum:
                        data_dict[TP][bs][test_name] = {
                            #"vLLM-requests/s":0, 
                            #"vLLM-tockens/s":0, 
                            "vLLM-Latency(sec)":0}

    elif args.customer == "Dell":
        input_output_enum = [[200, 1], [2000, 1], [7000, 1], [200,200], [200,1000], [2000, 200], [7000, 1000]]
        batch_enum = ["BS_" + str(bs) for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]]
        TP_type = ["TP_" + str(tp) for tp in [8]]

        for TP in TP_type:
            data_dict[TP] = dict()
            for bs in batch_enum:
                data_dict[TP][bs] = dict()

        # Fill in json
        for input_len, output_len in input_output_enum:
            test_name = "input_len_" + str(input_len) + "_output_len_" + str(output_len)
            for TP in TP_type:
                for bs in batch_enum:
                    data_dict[TP][bs][test_name] = {
                        "vLLM-Latency(sec)":0}


    # Convert the dictionary to a JSON-formatted string
    json_str = json.dumps(data_dict, indent=4)  # `indent=4` adds pretty-printing
    file_path = args.json  # Adjust to your desired file path
    # Save the JSON-formatted string to a file
    with open(file_path, 'w') as file:
        json.dump(data_dict, file, indent=4)  # `json.dump()` writes directly to a file



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--customer', choices=["SNOW", "ServiceNow", "Dell"], default="Dell", help="SNOW and ServiceNow are the same.")
    parser.add_argument('--json', default="benchmark.json", help="The filename of generated json")
    args = parser.parse_args()
    CreateJson(args)



'''
python create_json.py --customer SNOW --json benchmark_SNOW.json
python create_json.py --customer Dell --json benchmark_Dell.json
'''


