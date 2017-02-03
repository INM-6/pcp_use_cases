import sys
import os
import numpy
# Simple python file that parses the information of a asset file performance run
# output the data ready for consumption 


def process_single_file(file_base):
    """
    Parse data from file.
    Clean and convert to python types
    calculate mean of interesting values
    return the list of (name, mean)
    """
    # Ugly create file name string   
    file_name = os.path.join(os.getcwd(),"pcp_use_cases","data", file_base + ".txt")
    out_name = file_base + ".dat"

    # TODO: Try catch on file open?
    fp_in = open(file_name, "r")

    collected_data = []
    current_run = None
    run_idx = 0
    for line in fp_in.readlines():
        # Make use of feature in the data to know when to start a new 'group'
        if line.strip() == "{}":
            # Always continue to nex line
            continue

        if line.strip()[0:6] == "#Total":
            if current_run == None:
                current_run = []
            # We have data to store
            else:
                #print current_run
                collected_data.append(current_run)
                current_run = []

        # Strip comment
        if line.strip()[0] == "#":
            continue

        # split at the seperator
        data = line.strip().split(",")

        # remove possible white space
        data = [entry.strip() for entry in data]

        # TODO: Eval without try catch
        percent = int(eval(data[0]))
        name = data[1].split(" ")[-1]  # name can have prepended crap, just take the last part
        count = int(eval(data[2]))
        time = float(eval(data[3]))

        # Save the data
        current_run.append((name, percent, time, count))


    # We now have parsed data Calculate the mean of time (percentage is not interesting)
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

    #################################################
    # Retrieve data from following files
    files_for_parse = [
       ["output_original",       "Original"                ],
       ["output_log_exp_cython", "Cython"               ],
       ["output_log_exp_approx", "Approximate"               ],
       ["3c_sum_optimized",      "Combine\noperations"   ],
       ["4b_logprecompute",      "Precompute"                   ],
       ["5b_float32",            "Consistent\nprecision"  ],
       ["knl_mpi_gpp",           "MPI on KNL",  "prob_method" ]
        ]

    ##########################################
    # The raw names in the files is sometimes bad, we need a mapping from
    # raw name to display name. But we can have repeated names
    # TODO: CARLOS you should give these sections the correct name!
    typeset_data = [       
    # raw_data_name, display_name, color 
    # First set is all in the asset_min
    ("import"   ,"init 1",""),
    ("init"                     ,"init 2","#6DECC5"),
    ("data"                     ,"init 3","#749faf"),
    ("intersection_matrix"      ,"init 4","#687592"),

    # Some calculation is significant in the optimized version
    ("prob_method"              ,"prep","#8FA4C9"),

    # in between function call, totally not significant
    ("init"                     ,"neighbours 1","#6DECC5"),
    ("flatten"                  ,"adjoin","#7788AA"),
    ("sort"                     ,"neighbours 3","#6DECC5"),
    ("step"                     ,"neighbours 4","#6DECC5"),
    ("_pmat_neighbors"          ,"neighbours 5","#6DECC5"),

    # Our function of interest

    # Green colors
    ("init"                     ,"init","#93EDAA"),
    ("diff"                     ,"diff","#65A375"),
    ("reshape"                  ,"jsf","#2B8041"),
    ("sum"                      ,"factor","#2B8061"),

    # Red color
    ("log_DU2"                  ,"log","#EB8571"),
    ("log_DU2_ones"             ,"ones","#E08571"),
    ("log_DU2_copy"             ,"copy","#f78571"),

    # Yelow
    ("prod_DU2"                 ,"prod","#B1B55B"),
    ("sum_DU2"                  ,"sum","#FFF002"),

    # orange
    ("log"                      ,"logadd","#B35110"),
    ("exp"                      ,"exp","#EBCC71"),

    # Blue color
    ("step"                     ,"tail","#4F75E8"),
    ("_jsf_uniform_orderstat_3d","final","#308bb5"),

    # Finalize, printing output, takes not time
    ("joint_probability_matrix" ,"exit","#ffffff"),
    ("mask_matrices"            ,"exit","#ffffff"),
    ("cluster_matrix_entries"   ,"exit","#ffffff"),
    ("extract_sse"              ,"exit","#ffffff")]

    ################################################################
    # Get data from files, parse, convert raw names, assign collors
    # calculate totals, create labels
    # Debugging option: show original name in plot
    rename = True

    data = []
    bar_totals = []
    for file_base in files_for_parse:
        # Get data from file
        raw_data_for_single_bar = process_single_file(file_base[0])


        data_for_single_bar = []
        # Keep a running tab of the total runtime
        bar_total_single_bar = 0.0

        # We need to keep count of name occurrences, they might not be unique 
        # after cleanup in parse stage
        name_seen = [False] * len(typeset_data)


        for entry in raw_data_for_single_bar:
          
            # HACKY THE HACK!!! 
            # I want to not print a certain entry
            skip_name = None
            
            if len(file_base) == 3:
                skip_name = file_base[2]

            bar_total_single_bar += entry[1]
            type_set = False

            # Loop tru the typeset_data looking for first occurance
            for idx, entry_typeset in enumerate(typeset_data):
                if entry[0] == entry_typeset[0] and not name_seen[idx]:
 
                    # We have a match
                    name_seen[idx] = True  
                    type_set = True 

                    # Hack skip display and remove total
                    if skip_name == entry[0]:
                        bar_total_single_bar -= entry[1]
                        break

                    if rename:
                        name = entry_typeset[1]
                    else:
                        name = entry[0]

                    #                           value,      display name   , color                  
                    data_for_single_bar.append([entry[1], name,
                        "#ffffff" if entry_typeset[2] is "" else entry_typeset[2] ])

                    break

            if not type_set:
                data_for_single_bar.append([entry[1], entry[0],
                        "#ffffff"])


        data.append(data_for_single_bar)
        bar_totals.append(bar_total_single_bar)


    ##########################################################################
    # The total runtime as the bar labels
    bar_labels = [str(int(round(entry))) for entry in bar_totals]


    ########################
    # Which entries should have emphasis? And what explode wedges should be drawn?
    # Find the index based on labels in data
    emphasis = [[[[14,14],[14,14]],[[18,18],[18,18]]],    
                [[[14,14],[14,14]],[[18,18],[18,18]]], 
                [[[11,14],[11,14]],[[19,19],[19,19]]],
                [[[14,14],[14,15]]],
                [[[12,12],[12,12]],[[17,18],[17,18]]],
                [None],
                [None],
                [None],
                [None]
                ]



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
    exp_barch_tp_set["box_size_text_cutoff"]=0.3
    #offset left label
    exp_barch_tp_set["explode_label_offset_left"]= 0.52
    exp_barch_tp_set["explode_label_v_offset_left"]= 0.1
    #offset right side label
    exp_barch_tp_set["explode_label_offset_right"]= 0.80

    exp_barch_tp_set["explode_label_text"]={"ha":'left',  "va":'center', "fontsize":15,
                          "style":'italic', "zorder":4}


    exp_barch_tp_set["bar_label_text"]={"ha":'left',  "va":'bottom', "fontsize":20, "zorder":4}

    exp_barch_tp_set["box_label_text"]={ "ha":'center', "fontsize":18, "va":'center', "zorder":4}
    ##############################
    # Create a figure get the axis
    

    f, ax = plt.subplots(figsize=(20, 10))   

    ############################
    # Main call to functionality
    casbar.cascaded_exploding_barcharts(ax, data, emphasis, bar_labels,
                                        "percentage")

    ##############################
    # Some additional makeup of the figure
    #plt.title("Total runtime and duration of steps \n for specific optimization stages", fontsize = 17)

    xticks = [x + 0.25 for x in range(len(files_for_parse))]
    ax.set_xticks(xticks)

    # Use the file names as xbar labels
    ax.set_xticklabels([x[1] for x in files_for_parse])
    ax.tick_params(axis='both', which='major', 
                    bottom='off',      # ticks along the bottom edge are off
                      top='off',         # ticks along the top edge are off
                   
                   
                   labelsize=20)

    ax.set_yticks([])
    ax.set_yticklabels([])



    plt.xlim( -.2, 6.7 )

    plt.ylim(-5,  110)
    plt.show()
    f.savefig('myfig.png', dpi=300)

