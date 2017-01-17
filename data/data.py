import sys
import os
import numpy
# Simple python file that parses the information of a asset file performance run
# output the data ready for consumption 

# open the correct file
def process_single_file(file_base):
   
    file_name = os.path.join(os.getcwd(),"pcp_use_cases","data", file_base + ".txt")
    out_name = file_base + ".dat"

    fp_in = open(file_name, "r")

    collected_data = []
    current_run = None
    run_idx = 0
    for line in fp_in.readlines():
        # Make use of feature in the data
        if line.strip() == "{}":
            if current_run == None:
                current_run = []
            # We have data to store
            else:
                collected_data.append(current_run)
                current_run = []
            # Always continue to nex line
            continue

        if line.strip()[0] == "#":
            continue

        # split at the seperator
        data = line.strip().split(",")
        # remove possible white space
        data = [entry.strip() for entry in data]

        percent = int(eval(data[0]))
        name = data[1]
        count = int(eval(data[2]))
        time = float(eval(data[3]))

        current_run.append((name, percent, time, count))


    # We now have preparsed data

    mean_and_var = []
    for entry in zip(*collected_data):
        name = entry[0][0]

        times = []
        percentage = []

        for run in entry:
            times.append(run[2])
            percentage.append(run[1])

        mean_and_var.append((name, numpy.mean(times), numpy.var(times)))

    return mean_and_var

if __name__ == "__main__":

    files_for_parse = [
        "output_original",
        "output_log_exp_cython",
        "output_log_exp_approx"]


    for file_base in files_for_parse:
        print process_single_file(file_base)
    
