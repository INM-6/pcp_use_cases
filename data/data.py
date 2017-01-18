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
        name = data[1].split(" ")[-1]  # name can have prepended crap, just take the last part
        count = int(eval(data[2]))
        time = float(eval(data[3]))

        current_run.append((name, percent, time, count))


    # We now have preparsed data

    mean = []
    for entry in zip(*collected_data):
        name = entry[0][0]

        times = []
        percentage = []

        for run in entry:
            times.append(run[2])
            percentage.append(run[1])

        mean.append((name, numpy.mean(times)))

    return mean

if __name__ == "__main__":

    files_for_parse = [
        "output_original",
        "output_log_exp_cython",
        "output_log_exp_approx",
        #"output_diff_optimized",  #step 3a
        #"3b_reshape_optimized"    #step 3c
        "3c_sum_optimized",
        #"4a_copy_optimized"
        "4b_logprecompute",
        "5b_float32"
        ]

    data = []
    bar_totals = []

    for file_base in files_for_parse:
        raw_data_for_single_bar = process_single_file(file_base)
        # We need to add colors, for now pick white?

        data_for_single_bar = []

        bar_total_single_bar = 0.0
        for entry in raw_data_for_single_bar:
            bar_total_single_bar += entry[1]
            data_for_single_bar.append([entry[1], entry[0], "#ffffff"])

        data.append(data_for_single_bar)
        bar_totals.append(bar_total_single_bar)


    

    ########################
    # Find the index based on labels in data

    emphasis = [[[[14,14],[14,14]],[[18,18],[18,18]]],    
                [[[14,14],[14,14]],[[18,18],[18,18]]], 
                [[[11,13],[11,13]],[[14,14],[14,14]],[[19,19],[19,19]]],
                [[[14,14],[14,16]]],
                [[[12,12],[12,12]],[[17,18],[17,18]],[[19,20],[19,20]]],
                [None],
                ]

    ##########################################################################
    # The total runtime as the bar labels
    bar_labels = [str(int(round(entry)))+"s" for entry in bar_totals]



    ######################################################################
    # 'calculate' the explosion labels from the data and the emphasis
    # First loop is over the bar chart pairs
    for idx, entry_list in enumerate(emphasis):

        # THis loop is over the possible multiple emphasis pairs
        for entry in entry_list:
            
            # Skip if the data is not correct
            if entry is None or entry[0] is None:
                continue

            # FIrst idx should be from data
            first_range = entry[0]
            second_range = entry[1]

            # skip none values
            if first_range[0] is None or first_range[1] is None:
                continue

            # Calculate  the sum of the emphasis left, we need +1 for correct 
            # range
            sum = 0

            for range_idx in range(first_range[0], first_range[1]+1):
                sum += data[idx][range_idx][0]

            # Some fine tuning of the figure: If the explosion is less then 
            # some value do not print
            minimum_height = 1
            # TODO: This should be as a function of the displayed size
            # The data height can change allot depending on the raw data
            if sum >= minimum_height:
                # Append the label
                entry[0].append(str(int(round(sum))))
            else:
                entry[0].append(None)


            # Calculate  the sum of the emphasis right
            sum = 0
            for range_idx in range(second_range[0], second_range[1]+1):
                sum += data[idx+1][range_idx][0]

            if sum >= minimum_height:
                # Append the label
                entry[1].append(str(int(round(sum))))
            else:
                entry[1].append(None)

    ###################################################################
    # Plotting the chart
    import cascadedexplodingbarcharts as casbar
    import matplotlib.pyplot as plt

    exp_barch_tp_set = casbar.exp_barch_tp_set

    ##############
    # Type setting
    # all type setting by adapting global_type_setting
    exp_barch_tp_set["exploding_line"] = {'color':'k', "ls":'--', "lw":1.0}
    # important settings: controll from what size bar labels are not drawn
    exp_barch_tp_set["box_size_text_cutoff"]=0.5
    #offset left label
    exp_barch_tp_set["explode_label_offset_left"]= 0.61
    #offset right side label
    exp_barch_tp_set["explode_label_offset_right"]= 0.90

    

    ##############################
    # Create a figure get the axis
    f, ax = plt.subplots()   

    ############################
    # Main call to functionality
    casbar.cascaded_exploding_barcharts(ax, data, emphasis, bar_labels, 
                                 "percentage")

    ##############################
    # Some additional makeup of the figure
    plt.title("Total runtime and duration of steps \n for specific optimization stages", fontsize = 17)

    xticks = [x + 0.2 for x in range(len(files_for_parse))]
    ax.set_xticks(xticks)

    # Use the file names as xbar labels
    ax.set_xticklabels(files_for_parse)

    ax.set_yticks([])
    ax.set_yticklabels([])

    plt.xlim( -.2, plt.xlim()[1] + .2 )
    plt.ylim(- plt.ylim()[1] / 15,  plt.ylim()[1] + plt.ylim()[1] / 15)
    plt.show()

