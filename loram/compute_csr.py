import re
import numpy as np
import sys

log_file = sys.argv[1]
pattern = re.compile(r"\|(.*?)\|.*?\|\s*acc\s*\|↑\s*\|([\d\.]+)\|\s*±\s*\|([\d\.]+)\|")

values = []
std_devs = []

with open(log_file, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            task_name = match.group(1).strip()
            acc_value = float(match.group(2))
            acc_stderr = float(match.group(3))
            values.append(acc_value)
            std_devs.append(acc_stderr)

if values and std_devs:
    mean_value = np.mean(values)
    combined_std_dev = np.sqrt(np.sum(np.array(std_devs) ** 2) / len(std_devs))
else:
    mean_value = 0.0
    combined_std_dev = 0.0

result_line = f"\n|Average     |      |      |     |acc     |   |{mean_value:.4f}|±  |{combined_std_dev:.4f}|\n"

with open(log_file, 'a') as file:
    file.write(result_line)

print("results are appended to eval_1shot.log")
