import os
import csv
import argparse
import collections
import numpy as np

def get_log_files(input_folder):
    result = []
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # log_dir = script_dir + "/rpd"
    files = os.listdir(input_folder)
    for file in files:
        if (
            file.endswith(".csv")
            and not file.endswith(".stats.csv")
        ):
            result.append(input_folder + "/" + file)
    return result


def AccumulateTotalTime(csv_file):
    with open(csv_file, "r") as fd:
        csv_reader = csv.reader(fd)
        table = dict()
        time = collections.defaultdict(int)
        freq = collections.defaultdict(int)
        total_time = 0
        header = next(csv_reader)
        line_idx = 1

        for idx, val in enumerate(header):
            table[val] = idx

        for line in csv_reader:
            line_idx += 1
            if len(line) > 0:
                try:
                    kernel_name = line[table["Name"]]

                    curr_time = int(line[table["TotalDuration"]])
                    total_time += curr_time
                except Exception as err:
                    print(f"Line {line_idx}: {line}")
                    print(f"{err=}, {type(err)=}")
        print(f"{csv_file} total time {total_time}")
        return total_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True) 
    args = parser.parse_args()
    
    files = get_log_files(args.i)
    print(files)

    all_total_time = []
    for csv_file in files:
        start_idx = csv_file.rfind("/") + 1
        all_total_time.append(AccumulateTotalTime(csv_file))

    np_array = np.array(all_total_time)
    max_index = np.argmax(np_array)  # Find the index of the maximum value
    print(f"{files[max_index]} takes the most time {np_array[max_index]}")


