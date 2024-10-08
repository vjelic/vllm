import json
import argparse

def GetNextLength(args):
    benchmark = "vLLM-requests/s" if args.benchmark == 't' else "vLLM-Latency(sec)"
    with open(args.json, 'r') as file:
        data_dict = json.load(file)  # Load JSON data into a dictionary
        TP_type = "TP_" + str(args.tp)
        for BS in data_dict[TP_type].keys():
            for test, value in data_dict[TP_type][BS].items():
                if value[benchmark] == 0:
                    bs = int(BS.replace("BS_", ""))
                    input_len = int(test.split('_')[2])
                    output_len = int(test.split('_')[5])
                    if input_len==11000 and bs >=16:
                        continue
                    print(bs, input_len, output_len, end='')
                    exit(0)
        print("Finish.")

def SaveJson(args):
    with open(args.json, 'r') as file:
        data_dict = json.load(file)  # Load JSON data into a dictionary
        TP_type = "TP_" + str(args.tp)
        BS = "BS_" + str(args.bs)
        print(f"result = {args.result}")

        if args.benchmark == 't': # result= "Throughput: 1088.98 requests/s, 70783.49 tokens/s"
            result = args.result.split(' ')
            requests = float(result[1])
            tokens = float(result[3])
            data_dict[TP_type][BS][f"input_len_{args.input_len}_output_len_{args.output_len}"]["vLLM-requests/s"] = requests
            data_dict[TP_type][BS][f"input_len_{args.input_len}_output_len_{args.output_len}"]["vLLM-tockens/s"] = tokens
        else: # result= "Avg latency: 34.692651053000006 seconds"
            if args.result == "OOM":
                data_dict[TP_type][BS][f"input_len_{args.input_len}_output_len_{args.output_len}"]["vLLM-Latency(sec)"] = "OOM"
            else:
                result = args.result.split(' ')
                latency = float(result[2])
                data_dict[TP_type][BS][f"input_len_{args.input_len}_output_len_{args.output_len}"]["vLLM-Latency(sec)"] = latency

    with open(args.json, 'w') as file:
        json.dump(data_dict, file, indent=4)  # `json.dump()` writes directly to a file
        print("Save results...")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=["t", "l"], required=True) # Throughput or laterncy
    parser.add_argument("--json", type=str, required=True) 
    parser.add_argument("--tp", type=int)
    parser.add_argument("--result", type=str)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    args = parser.parse_args()

    if args.result == None:
        GetNextLength(args)
    else:
        SaveJson(args)

# For dev:
# GetNextLength cmd
#   python JsonHelper.py --benchmark l --json benchmark_SNOW.json --tp 2
# SaveJson cmd:
#   python JsonHelper.py --benchmark l --json benchmark_SNOW.json --tp 2 --result "Avg latency: 34.692651053000006 seconds" --bs 1 --input-len 4400 --output-len 300


