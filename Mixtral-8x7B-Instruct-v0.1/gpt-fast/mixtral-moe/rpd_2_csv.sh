#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -i input_dir"
    exit 1
}

# Parse command-line arguments
while getopts ":i:" opt; do
    case ${opt} in
        i )
            input_dir=$OPTARG
            ;;
        \? )
            usage
            ;;
    esac
done

# Check if input_dir is set
if [ -z "$input_dir" ]; then
    usage
fi

# Loop over the range 0 to 7
for i in {0..7}
do
    # Construct the file names with directory paths
    input_file="${input_dir}/trace_${i}.rpd"
    output_file="${input_dir}/trace_${i}.csv"

    echo "input_file = ${input_file}"
    
    # Execute the SQLite command
    sqlite3 $input_file <<EOF
.mode csv
.header on
.output $output_file
SELECT * FROM top;
.output stdout
EOF

done
