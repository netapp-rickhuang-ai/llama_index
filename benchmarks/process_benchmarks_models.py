import os
import re
import json
import argparse

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Process benchmark model files.")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'llama3.1')")
args = parser.parse_args()
model = args.model

# Define the folder containing the benchmark results
benchmark_folder = "/Users/yinray/Documents/workspace-mac/llama_index/benchmarks"

# Define the pattern to match file names like "{model}.txt" or "{model}.txt.1"
file_pattern = re.compile(rf"{re.escape(model)}\.txt.*")

# Define a pattern that captures the metric name and its numerical value regardless of its unit.
metric_pattern = re.compile(r"^(prompt eval count|prompt eval rate|eval rate):\s+([\d.]+)")

# Initialize dictionary of lists to store values for each metric
tokens_per_second_values = {"prompt eval count": [], "prompt eval rate": [], "eval rate": []}

# Iterate through the files in the benchmark folder
for filename in os.listdir(benchmark_folder):
    if file_pattern.match(filename):
        file_path = os.path.join(benchmark_folder, filename)
        with open(file_path, "r") as file:
            for line in file:
                # Search for the metric and numerical value in the line
                match = metric_pattern.search(line)
                if match:
                    metric = match.group(1)
                    value = float(match.group(2))
                    tokens_per_second_values[metric].append(value)
                    print(f"\n[INFO] Processed {metric}: {value:.2f}")

# Initialize variables to store the results
tokens_per_second_values_out = {"avg prompt eval count": [], "avg prompt eval rate": [], "avg eval rate": []}

# Calculate and print the average for each metric
for metric, values in tokens_per_second_values.items():
    if values:
        average = sum(values) / len(values)
        print(f"Average {metric}: {average:.2f}")
        tokens_per_second_values_out[f"avg {metric}"].append(average)
    else:
        print(f"No {metric} values found in the specified datasets.")

# Print final results
print("\nFinal Results:")
for result, values in tokens_per_second_values_out.items():
    print(f"{result}: {values}")

# Store the results into a JSON file

# Update these variables as needed
# model = "llama3.1"
version = "latest"
no_of_params = "8B"

output_filename = f"{model}.{version}.{no_of_params}_results.out.json"
with open(output_filename, "w") as json_file:
    json.dump(tokens_per_second_values_out, json_file, indent=4)

print(f"\nResults saved to {output_filename}")
