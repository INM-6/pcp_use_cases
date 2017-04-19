import numpy
import matplotlib.pyplot as plt
import copy
from matplotlib.path import Path
import matplotlib.patches as patches

"""
# cascaded_exploding_bar_chart
Create a cascade of two or more exploding bar-charts from data with controllable typesetting.

![See graphic below for an rough example](/example.png?raw=true "") 

Fully typeset example [1]
![See graphic below for an rough example](/full_example.png?raw=true "") 

## Features
1. Size of bars in, and the total number of bar-charts is generated from data
2. height of bars based on raw data, normalized or percentage (each bar normalized individually)
3. Option to display a relative size bar when normalizing the data
4. Show multiple labeled relationships between stacks
5. Explosion lines to show relation between stacked bars
6. The explosion wedge can have labels and a shaded background
7. Stacked bars and boxes can have labels
8. Box labels will not display when text is too big for the box (controllable)
9. Only matplotlib needed
10. Most textual and graphical elements can be typeset without changing the code
    
## Usage
Call cascaded_exploding_barcharts() with your data. If you want to typeset you
have to do this before the call.

## Input arguments cascaded_exploding_barcharts

    ax:     Axes object to plot the figure on (allows embedding in more complex 
            figures)

    data:   The input data, 3d array: A list of bar-chart data
            where each bar-chart is a list of triplets containing:
            [value, label, #color]
            See run_example() for detailed example of data structure

    emphasis: The entries in the bar-chart to emphasize and explode from and TO
            It is a 3d array: A list of triples, setting the range and the label
            The first range gives the emphasis in the source bar stack
            The second range gives the target for the explosion lines (starting
            from the emphasized range)
            e.g. To create the explosion lines in the graphical example you
            would use range: [[1,2,"EL"],[1,2,"EL"]] 
            Use None for a tripled pair not wanting emphasis 

    bar_labels: An optional label to be printed above each stack bar-chart
            (BL in the graphical example)

    representation: string
            "None": display data without normalizing (could results in 
                    different height bars!)
            "normalized": normalize bar height
            "percentage": normalize and multiply with 100 for %
            
            This settings has influence on the y-axis labels and how the 
            label printing is controlled. Also offset might needs to be adapted

## Typesetting

Typesetting is controlled by changing the variables in the global variable
 'exp_barch_tp_set' (A global,I know, I know).
Default typesetting is for the default matplotlib figure size and of a simple nature
changing the typesetting should be done before calling cascaded_exploding_barcharts()

    Control features (True/False):
        box_label: Print the labels in the boxes 
        bar_label: Print a label above each stack
        explode_label: Print labels between the explode lines
        explode_bg: Give the wedge between the explosion lines a color

    box_size_text_cutoff: Controls the automatic check if text is to big for a
        box (Default looks ok for standard matplotlib figure size).

The other options are either text offsets -or-
dictionaries forwarded to either axes.bar axes.line or axes.text functions 
(see matplotlib documentation for more information). 

The "zorder" is used to correctly stack the different graphical elements. It
is not advisable to change these settings.

It is possible to use named color names instead of the #rgb used currently for 
typesettings EXCEPT the bar color. The gradient method for the border
is simplistic (borderline buggy / broken) and will fail for non #rgb entries.

## Graphical example
```              
               BL           BL
            _________     _________ 
            | 3     |     | 3     |   
            |=======|_ _ _|=======|  
            | 2     |...EL|=2=====| 
            |       |.....|=1=====|
            |       |EL../| 0     |
            |=======|.../ |       |
            | 1     |../  |       |
            |       |./   |       |
            |=======|/    |       |
            | 0     |     |       |
            |_______|     |_______|
```

## TODOS
1. bar_labels must always be supplied even when turned off (can be None)
2. Major loop contains an ugly minus 1
3. Do something with the global typesetting dict exp_barch_tp_set
4. Test all corner cases of None types in emphasis
5. Check if raw plotting wedges are correct
6. It is possible to use a 'darkened' color for the box border. Lighting
   should also be possible, but rgb is hard

## References:
[1] "ASSET for JULIA: executing massive parallel spike correlation analysis on a KNL cluster";
  Carlos Canova, Wouter Klijn, Et. Al. ;HBP student conference; 2017
   
Keywords: python matplotlib exploding exploded barchart barcharts bar-chart bar-charts cascaded cascading 
"""


def run_example():
    #############################
    # Define some data to display
    data = [[[2,"foo_1", "#3F8080"],[2,"foo_2", "#346080"], [3,"foo_3", "#30A280"],
             [4,"bar_1", "#CFA080"], [4,"bar_2", "#C08080"], [5,"bar_3", "#CB8060"]],
            [[2,"foo_1", "#3F8080"],[2,"foo_2", "#346080"], [3,"foo_3", "#30A280"],
             [0.4,"bar_1", "#CFA080"], [0.4,"bar_2", "#C08080"], [0.5,"bar_3", "#CB8060"]],
            [[2,"foo_1", "#3F8080"],[.2,"foo_2", "#346080"], [.3,"foo_3", "#30A280"],
             [0.4,"bar_1", "#CFA080"], [0.4,"bar_2", "#C08080"], [0.5,"bar_3", "#CB8060"]]]

    emphasis = [[[[1,2,"30"],[1,2,"60"]], [[4,5,"30"],[4,5,"10"]]],  #2 wedges with emphasis 
                [[[1,2, "40"],[1,2, "10"]],],  
                [None,],]    
    
    bar_labels = ["140", "90", "60"]

    ##############
    # Type setting
    # all type setting by adapting global_type_setting
    exp_barch_tp_set["exploding_line"] = {'color':'k', "ls":'--', "lw":1.0}
    # important settings: controll from what size bar labels are not drawn
    exp_barch_tp_set["box_size_text_cutoff"]=0.6

    ##############################
    # Create a figure get the axis
    f, ax = plt.subplots()   

    ############################
    # Main call to functionality
    cascaded_exploding_barcharts(ax, data, emphasis, bar_labels, 
                                 "percentage")

    ##############################
    # Some additional makeup of the figure
    plt.title("Example of cascading exploding bar-charts", fontsize = 17)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["First", "Second", "Third"])

    ax.set_yticks([])
    ax.set_yticklabels([])

    plt.xlim( -.2, plt.xlim()[1] + .2 )
    plt.ylim(- plt.ylim()[1] / 15,  plt.ylim()[1] + plt.ylim()[1] / 15)
    plt.show()
   
   
#Default types settings    
# TODO: change from global to something else
exp_barch_tp_set = {
    # Border types for the bar boxes
    "emphasis_box":{ "width":0.5, "lw":2.5, "zorder":3},
    "normal_box":{ "width":0.5, "lw":1, "zorder":2},

    # text labesl in the bar boxes
    "box_label":True,
    "box_label_text":{ "ha":'center',  "va":'center', "zorder":4},
    # controls the when drawing is skipped
    "box_size_text_cutoff":1.0,  
    # offset of text in the box
    "box_label_offset":0.2, 
    # Box border gradient of the box-color, use 0.0 for no black
    "box_border_gradient":0.2,

    # label above the bars
    "bar_label": True,
    "bar_label_text":{"ha":'left',  "va":'bottom', "fontsize":15, "zorder":4},
    # Offset of text from left side of bars
    "bar_label_offset": 0.15,

    # The relative bar settings
    "relative":True,
    "relative_box":{ "width":0.04, "lw":0, "zorder":5, "hatch":"-", 
                    "edgecolor":"#ffffff"}, # The color with - is not displayed correctly. always grayish
    "relative_box_h_offset": -0.005,

    # Line type for the lines between the bars
    "exploding_line":{'color':'k', "ls":'--', "lw":1.0, "zorder":4},
    # Label between the explosion lines
    "explode_label": True,
    "explode_label_text":{"ha":'left',  "va":'center', "fontsize":13,
                          "style":'italic', "zorder":4},
    #offset left label
    "explode_label_offset_left": 0.52,
    # Correction in vertical position (used when the wedges are skewed up or down)
    # v_offset controlls the magnitude of the correction 0 = no correction
    "explode_label_v_offset_left": 0.15, 

    #offset right side label
    "explode_label_offset_right": 0.83,
    "explode_label_v_offset_right": 0.6,

    "explode_bg":True,
    "explode_bg_xs_offset":0.009,
    "explode_bg_ys_offset":0.0013,
    "explode_bg_vert":{"facecolor":"#eeeeee", "lw":0, "zorder":1}

    } 

def convert_color(color, multiplicateion_factor=1.0):
    """
    Helper function transforms a #AABBCC color to a different color by
    multiplying with a factor   

    There are probably better ways but
    NO ERROR CHECKING, use at your own risk!!
    """

    new_color = "#"

    # Add mutated first rgb
    for first, second in [(1,2), (3,4), (5,6)]:

        color_hex = "0x" + color[first] + color[second]
        color_changed =  hex(int(int(color_hex, 16) * multiplicateion_factor))

        if color_changed > hex(0xFF):
            color_changed = 0xFF


        if len(str(color_changed)[2:4]) == 1:
            new_color += "0"
        new_color += str(color_changed)[2:4]

    return new_color

def create_bar_chart_with_emphasis(ax, data, emphasis = None,
                                   bar_label = None,
                                   bar_sum = None,
                                   stack_idx = 0):
    """
    Create a stacked bar-chart with some bars emphasised and a lable in the box
    data: list of [value, label, #color]
    emphasis: pair of boxes to emphasize
    bar_label: text to print above the stack
    stack_idx: Which stack we are working on
    """
    pos_sum = 0.
    bar_dict = {}   # Needed for getting the location of the emphasis 

    if exp_barch_tp_set["box_label"]:
        # Ugly hack but needed to get the size of the printed text,
        # But mhe.. who cares JOLO!!!
        renderer = ax.figure.canvas.get_renderer()

    # If we have data in the emphasis array get the ranges
    # We can have multiple ranges, create a list of emphasis boxes
    emph_boxes = []
    if emphasis and emphasis[0]:   
        emph = True
        # we can have one or more emphasis description
        for emph_entry in emphasis:
            # plus one to get inclusive range

            for idx in range(emph_entry[0][0], emph_entry[0][1] + 1):
                emph_boxes.append(idx)
    else:
        emph = False


    # The boxes are drawn from bottom to top. This causes emphasis box top
    # line to be overdrawn of there are more written after. Save the top fat box
    # and redraw
    box_to_redraw = None

    for idx in range(len(data)):
        (value, label, color) = data[idx]
        box_settings = exp_barch_tp_set["normal_box"]
        # If we are drawing with emph and we are in the range
        if emph and idx in emph_boxes:
                box_settings = exp_barch_tp_set["emphasis_box"]


        edge_color = convert_color(color, 
                          exp_barch_tp_set["box_border_gradient"])

        # Draw the bar, save it because we might need to redraw

        last_bar = ax.bar(stack_idx, value, bottom=pos_sum, color=color, 
                    align='edge', edgecolor=edge_color,
                    label=label, **box_settings)

          
        if exp_barch_tp_set["box_label"]:
            text_var = ax.text(stack_idx + exp_barch_tp_set["box_label_offset"],
                               pos_sum + 0.5 * value,
                    label, **exp_barch_tp_set["box_label_text"])

            # Now check the size of the printed text (can only be done
            # after drawing)                               
            text_height = \
                text_var.get_window_extent(renderer=renderer).height
            # Remove if larger then some user controlled size
            box_size_text_cutoff = exp_barch_tp_set["box_size_text_cutoff"]

            if (text_height * box_size_text_cutoff) > value:
                text_var.remove()

        pos_sum += value

    # Add the label of the bar (if there)
    if exp_barch_tp_set["bar_label"] and bar_label:
        text_var = ax.text(stack_idx + exp_barch_tp_set["bar_label_offset"],
                          pos_sum, bar_label,
                          **exp_barch_tp_set["bar_label_text"])

    # Add the relative size bar
    # TODO: Make selectable
    if exp_barch_tp_set["relative"]:
        ax.bar(stack_idx + exp_barch_tp_set["relative_box_h_offset"],
           bar_sum, bottom=0, color='k', 
                    align='edge', 
                    label=label, **exp_barch_tp_set["relative_box"])


def explosion_line_y_top_and_bottom(data, emphasis):
    """
    Helper function that returns the top and bottom y locations based on the
    data en the emphasized range supplied for a SINGLE bar-chart
    """
    top = 0.
    bottom = 0.   

    if emphasis is None:
        return   # early exit, but no return because, there might be emph on right

    for i, (value, label, color) in enumerate(data):           
        top += value  # We need the top so always add  
        # If we have found the data index for the top box
        if emphasis[1] is i:    # If none it will never match and just
            break # fall out the loop

        # Now bottom box
        if emphasis[0] is None:  # Take the bottom ==> id == 0
            bottom = 0.
        if i < emphasis[0]:      # If we are not yet at the bottom box
            bottom += value
        else:
            pass  # We are done, we need the bottom so no adding of value

    return top, bottom


def explosion_line_y_points(data, data2, emphasis):
    """
    Helper function that returns the top and bottom y locations based on the
    data en the emphasized range supplied for a left and right bar-chart

    emphasis contains two list with boxes that have the emphasis 
    The lines should go to the bottom of the first and the top of the second
    left and right
    [[1,3],[1,3]]
    The lists could also be None both at all levels
    """

    y_begin_bottom_line = 0.
    y_begin_top_line = 0.
    y_end_bottom_line = 0. 
    y_end_top_line = 0.

    # Early exit with valid value when no emphasis found
    if emphasis is None:
        return [y_begin_bottom_line, y_end_bottom_line], \
           [y_begin_top_line, y_end_top_line]


    y_begin_top_line, y_begin_bottom_line = explosion_line_y_top_and_bottom(
        data, emphasis[0])

    y_end_top_line, y_end_bottom_line = explosion_line_y_top_and_bottom(
        data2, emphasis[1])

    return [y_begin_bottom_line, y_end_bottom_line], \
           [y_begin_top_line, y_end_top_line]


def explosion_line_x_points(stack_idx=0):
    """
    Helper function to create x-location based on the stack_idx
    """
    return [stack_idx + 0.50, stack_idx + 0.99], \
           [stack_idx + 0.50, stack_idx + 0.99]


def display_explosion(ax, data, emphasis, representation, chart_id=0):
    """
    Helper function to draw the explosion lines and labels
    """
    if emphasis is None:
        return

    # The start and endpoints of the exploding line depends on the data
    ys_bottom_line, ys_top_line = explosion_line_y_points(
                            data[chart_id], data[chart_id+1], emphasis)
    xs_bottom_line, xs_top_line = explosion_line_x_points(chart_id) 

    ax.plot(xs_bottom_line, ys_bottom_line, 
            **exp_barch_tp_set["exploding_line"])        
    ax.plot(xs_top_line, ys_top_line, 
            **exp_barch_tp_set["exploding_line"])

    # Add text at the centre to show the size of the 'sum'
        # Add the label of the bar (if there)
    if exp_barch_tp_set["explode_label"]:
        # The vertical location of the label might be a little offset if the 
        # wedge has a large vertical shift to the next bar
        # Use the explode_label_offset combined with the ys_top_line and bottom
        # line to create an interpolation location which is better
        mid_left = (ys_top_line[0] + ys_bottom_line[0]) / 2
        mid_right = (ys_top_line[1] + ys_bottom_line[1]) / 2
        # Calculate side and direction of shift between bards
        midline_dx = mid_right - mid_left

        # left side of 
        left_cor = exp_barch_tp_set["explode_label_v_offset_left"]
        left_offset = exp_barch_tp_set["explode_label_offset_left"] 
        # Calculate some fraction based on the offset and the width
        fraction_away_from_left = left_offset / .50
        # magnitude correction * fraction times the dx = correction
        correction_left = left_cor * fraction_away_from_left * midline_dx
        if not emphasis[0][2] is None:
            ax.text(chart_id + exp_barch_tp_set["explode_label_offset_left"], 
                mid_left + correction_left, 
                emphasis[0][2],
                  **exp_barch_tp_set["explode_label_text"])


        # Detail explanantion can be found in left side of correctopm
        right_cor = exp_barch_tp_set["explode_label_v_offset_right"]
        right_offset = exp_barch_tp_set["explode_label_offset_right"] 
        fraction_away_from_right = (1 - right_offset ) / .50  
        correction_right = - right_cor * fraction_away_from_right * midline_dx
        if not emphasis[1][2] is None:
            ax.text(chart_id + exp_barch_tp_set["explode_label_offset_right"], 
                mid_right  + correction_right, 
                emphasis[1][2],
                  **exp_barch_tp_set["explode_label_text"])
    
    # Draw a gray background between the explosion lines
    if exp_barch_tp_set["explode_bg"]:
        # If we are doing work in percentage its easier to have the y_offset be corrected
        if representation == "percentage":
            offset_correct = 100.
        else:
            offset_correct = 1.
        # There are some small plotting issues, depending on the representation and size 
        # of the window, these can be corrected here
        xs_offset = exp_barch_tp_set["explode_bg_xs_offset"] #0.009
        ys_offset = offset_correct * exp_barch_tp_set["explode_bg_ys_offset"] #0.02
        # Convert the locations we have to verts
        verts = [
            (xs_bottom_line[0] + xs_offset, ys_bottom_line[0] + ys_offset), # left, bottom
            (xs_top_line[0] + xs_offset, ys_top_line[0]), # left, top
            (xs_top_line[1] + xs_offset, ys_top_line[1]), # right, top
            (xs_bottom_line[1] + xs_offset, ys_bottom_line[1] + ys_offset), # right, bottom
            (xs_bottom_line[0]+ xs_offset, ys_bottom_line[0] + ys_offset), # ignored
            ]
        # plot the patch
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)    
        patch = patches.PathPatch(path, **exp_barch_tp_set["explode_bg_vert"]) #facecolor='orange', lw=0, zorder=1
        ax.add_patch(patch)
      
def normalize_or_percentage_data(data, representation=None):
    """
    Normalization or percentage version of the input data
    Returns the total bar size (possibly normalized) for relative bar plotting
    """
    
    bar_sums = []
    first_bar_sum = None
    for bar in data:
        #Calculate the sum (also needed for the relative bar length)
        sum = 0.
        for entry in bar:
            sum += entry[0]

        # If we are processing the first bar
        if first_bar_sum is None:
            first_bar_sum = sum

        if not representation is None:
            #now normalize or percentage
            size_bar = 1.0
            if representation == "percentage":
                size_bar = 100.0  # %

            norm_sum = None
            for entry in bar:
                    norm_sum = (entry[0] / sum ) * size_bar
                    entry[0] = norm_sum

            bar_sums.append((sum / first_bar_sum) * size_bar )
        else:
            bar_sums.append(sum)

    return bar_sums

def cascaded_exploding_barcharts(ax, data, emphasis, bar_labels,
                                 representation=None):
    """
    Insert a cascaded exploding barchart into ax

    ax:     Axes object to plot the figure on (allows embedding in more complex 
            figures)
    data:   The input data, 3d array: A list of bar-chart
            where each bar-chart is a list triplet containing:
            [value, label, #color]
            See run_example for detailed example

    emphasis: The entries in the bar-chart to emphasize and explode from and TO
            3d array: A list of paired ranges
            The first range gives the emphasis in the source stack
            The second range gives the target for the explosion lines (starting
            from the emphasized range
            e.g. To create the explosion lines in the graphical example you
            would use paired range: [[1,2,"left"],[1,2,"right"]] 

    bar_labels: An optional label to be printed above each stack bar-chart
            (EL in the graphical example)



    representation: None, display data without normalizing (could results in
            different hight bars!)
            normalized: normalize bar hight
            percentage: normalize and multiply with 100 for %
            Has influence on the y-axis labels and how to label printing is 
            controlled.

    See sourcefile desciption for detailed explanation
    """
    # Get a deep copy to allow mutations on the data for normalization
    data_internal = copy.deepcopy(data)
    bar_sums = normalize_or_percentage_data(data_internal, representation)

    # First bar is created outside of the loop, because we to explode n-1 times


    create_bar_chart_with_emphasis(ax, data_internal[0], emphasis[0], bar_labels[0], bar_sums[0],
                                   0)

    # Todo this -1 is ugly!! but needed for the explosion lines: I like the
    # location better at this place in the loop
    for idx in range(len(data_internal)-1):    
        for emphasis_subset in emphasis[idx]:
            display_explosion(ax, data_internal, emphasis_subset,
                              representation, chart_id=idx)


        create_bar_chart_with_emphasis(ax, data_internal[idx+1], 
                emphasis[idx+1], bar_labels[idx+1], bar_sums[idx+1],
                idx+1)
        

if __name__ == "__main__":
    run_example()

    