import time
import math
import sys
import os

"""
# simplepytimer
Measure the duration of python code by creating an timer objects with only a name.

Two simple time measurement classes:
Timer is single entrant and measure time since last occurrence.
MultiTimer remembers the name for a measure point and can be used in loops.

Display results by calling a class function

Not thread save, but if you want to do performance tests for such a program,
you should not be looking at a timer with 'simple' in it's name.

```python
import time
from simplepytimer import MultiTimer

MultiTimer("start")
time.sleep(0.1)
MultiTimer("end")

MultiTimer.print_timings()
```

## Features
1. Construct the object with only a string and create a measure point
2. Optional jump in level for displaying results
3. Optionally exclude most of the timer work in the timing results
4. Optionally export to file (csv, standard), with # comments
5. Reset internal state to Null
6. Only standard imports
7. Self test when running the python file.

## Usage
Construct two timer object with a name to start a measurement. Construct more
to get durations between the timer.
call print_timings to get the results on the screen

See run_example() for a detailed use-cases with line by line comments

### MultiTimer(name, level=0 [, exclude_timer_work=False])
    Constructor initializes a measure point. You need a minimum 
    of two to measure a stretch of time.
    
    name:  string identifier, printed in the results
  
    level: how much to jump in when printing
  
    [exclude_timer_work]: bool option available for the MultiTimer. Do not
         measure some calculations internal in the timer. The timing might be
         a little better but the total duration of the programm will no be the 
         same anymore as measured in the timer. 

### @classmethod reset()
    Clears the timer as if not being seen before.
    
     
### @classmethod print_timings(cls, header=True, seperator=" : ", prefix="  ")
    Display the results currently stored in the timer (can be called multuple times)

    header: Print a header with info of metrics before printing results

    seperator: seperator between metrics

    prefix:  What character to print before name to jump in.

### @classmethod to_file_like_as_csv(cls, fp, header=True)
    Print 
    
    fp:   'file like' object to send output data to
    
    header: Print explaining header

# run_example output
```
#Total run time time (sec)  :  1.40700006485
#% : name : count : total time spend (sec)
  0 : Start : 1 : 0.0
  7 : Second timer : 1 : 0.101000070572
  7 :   Third timer, with a jump in : 1 : 0.0999999046326
 14 : Fourth timer : 2 : 0.200999975204
 71 : Fifth timer seen 10 times! : 10 : 1.00500011444

 Second timing results:
#Total run time time (sec)  :  1.21000003815
#% : name : count : total time spend (sec)
  0 : Start : 1 : 0.0
  0 : loop : 7 : 0.000999927520752
  0 :       Timer 1 : 1 : 0.0
  0 :       Timer 10 : 10 : 0.0
  0 :       Timer 100 : 100 : 0.0
  0 :       Timer 1000 : 1000 : 0.00500011444092
  3 :       Timer 10000.0 : 10000 : 0.0319998264313
 10 :       Timer 100000.0 : 100000 : 0.117000102997
 87 :       Timer 1000000.0 : 1000000 : 1.05500006676

 Timing of the timer:
#Total run time time (sec)  :  1.21600008011
#% : name : total time spend (sec)
  0 : start : 0.0
  0 : Timer 1 : 0.0
  0 : Timer 10 : 0.0
  0 : Timer 100 : 0.0
  0 : Timer 1000 : 0.00500011444092
  3 : Timer 10000.0 : 0.0319998264313
 10 : Timer 100000.0 : 0.118000030518
 87 : Timer 1000000.0 : 1.06100010872
Press any key to continue . . .
```

# TODOS
1. Timer is not as feature complete as MultiTimer
2. Better function doc strings
3. Refactor out common print capability
4. Decorator?
5. Python 3?

Keywords: python timer timing timeit performance runtime duration

"""

class Timer:
    """
    A very simple timer class 

    Construct a the class to create a measure point. The supplied name will
    be stored including the time the timer was hit

    The measured time is the elapsed time since the previous measure point.
    """
    timers = []

    def __init__(self, name, level=0):

        Timer.timers.append((name, time.time(), level))

    @classmethod                       
    def reset(cls):
        """
        Reset the class 
        """
        MultiTimer.timers = []

    @classmethod
    def print_timings(cls, header=True, seperator=" : ", prefix="  "):
        timers = Timer.timers
        total = 0.0

        # Calculate the total runtime
        first = timers[0][1]

        # Loop over all the remaining timers to calculate the total
        # Plus one for the start time
        for idx in range(1, len(timers)):
            total += timers[idx][1] - timers[idx-1][1]

        print ("#Total run time time (sec) {0} {1}".format(seperator, total))

        # TODO: THe # is right aligned to the largest number, thus when printing
        # without separator there are extra spaces at the start of a line....
        print ("#%{0}name{0}total time spend (sec)".format(seperator))

        # init the previous time with the first timer
        previous_time = timers[0][1]
        for timing in timers:
            # For pretty printing use the level to jump in
            spaces = prefix * timing[2] 

            print ("{0:>3}{3}{1}{3}{2}".format(
                int(round(100 * ((timing[1] - previous_time) / total))),  # %
                spaces + timing[0],                         # spacer
                timing[1] - previous_time,                  # time in sec
                seperator))                                 # make up
            
            # Save the timing so we can take the difference
            previous_time = timing[1]

    @classmethod
    def to_file_like_as_csv(cls, fp, header=True):
        """
        Print a cvs representation to a file like object. Default a header is
        print at the first line
        """
        try:
            # Temporary reroute the stout to a file
            orig_stdout = sys.stdout
            sys.stdout = fp

            # Print the results
            MultiTimer.print_timings(header=True, seperator=",", prefix="")

        finally:
            # Always put back the original stout
            sys.stdout = orig_stdout


class MultiTimer:
    """
    A Simple timer class that keeps count of the number of times it is hit

    Construct a the class to create a measure point. The supplied name will
    be used internally to identify the timer. An optional level can be used 
    to visually distinguish timers (when they are in a loop for instance.)

    The measured time is the elapsed time since the previous measure point.
    """
    # A set of class level variables, shared by all objects of this type!
    timers = {}
    count = {}
    order = []
    level = {}

    # Time elapses since the last init call
    last = None

    def __init__(self, name, level=0, exclude_timer_work=False):
        """
        MultiTimer init, call with a name and optional level to create a measure
        point
        
        arguments:
        name: (String) identifier for your measure point. For most use case 
               this should be unique (but you could reuse a id).
        level: Set how much the measurepoint is jumped in when printing the 
               results. Could be used for timers in functions, effect is only
               visual.

        exclude_timer_work: Reduces the amount of 
        """
        current = time.time()

        # check if we have seen the timer id before
        if name in MultiTimer.timers:    # This might be the slowest part of the code     
            # If yes, calculate time since last timer
            MultiTimer.timers[name] = MultiTimer.timers[name] + \
                (current - MultiTimer.last )
            # increase the counter
            MultiTimer.count[name] += 1

        # If it is a new timer id.
        else:  
            # Check if we are the first ever.
            if not MultiTimer.last:
               MultiTimer.last = current # set time to now

            # Save the id and order
            MultiTimer.order.append(name)

            # initialize / save the data point
            MultiTimer.timers[name] = 0.0 + (current - MultiTimer.last)
            MultiTimer.count[name] = 1
            MultiTimer.level[name] = level

        # Save the last timer ( this means that we also measure the time it
        # takes for the timer functionality)
        # exclude_timer_work we take the current time (reduces the influence 
        # of the Timer code of the measurement a little
        if exclude_timer_work:
            MultiTimer.last = time.time()
        else:
            MultiTimer.last = current
     
    @classmethod                       
    def reset(cls):
        """
        Reset the class 
        """
        MultiTimer.timers = {}
        MultiTimer.count = {}
        MultiTimer.order = []
        MultiTimer.level = {}

        # Do not forget to reset the last measure point to None
        MultiTimer.last = None

    @classmethod
    def print_timings(cls, header=True, seperator=" : ", prefix="  "):
        """
        Print a textual overview of the timing results
        """
        timers = cls.timers
        count = cls.count
        total = 0.0
        # Calculate the total runtime
        for key in cls.order:
            total += timers[key]

        print ("#Total run time time (sec) {0} {1}".format(seperator, total))

        # TODO: THe # is right aligned to the largest number, thus when printing
        # without separator there are extra spaces at the start of a line....
        print ("#%{0}name{0}count{0}total time spend (sec)".format(seperator))       
        for key in cls.order:
            # For pretty printing use the level to jump in
            spaces = prefix * MultiTimer.level[key] 

            print ("{0:>3}{5}{1}{2}{5}{3}{5}{4}".format(
               int(round(100 * (timers[key] / total))),
               spaces,
               key, 
               count[key], 
               timers[key],seperator))

        

    @classmethod
    def to_file_like_as_csv(cls, fp, header=True):
        """
        Print a cvs representation to a file like object. Default a header is
        print at the first line
        """
        try:
            # Temporary reroute the stout to a file
            orig_stdout = sys.stdout
            sys.stdout = fp

            # Print the results
            MultiTimer.print_timings(header=True, seperator=",", prefix="")

        finally:
            # Always put back the original stout
            sys.stdout = orig_stdout


def run_example():
    """
    Run a small example program which doubles as a performance test
    """
    ###############################################################
    # Example how to use the MultiTimer
    # create a multitimer by simple creating the class with a string
    MultiTimer("Start")
    time.sleep(0.1)
    MultiTimer("Second timer")
    time.sleep(0.1)
    # you can add a level for clarity
    MultiTimer("Third timer, with a jump in", 1)
    time.sleep(0.1)
    # If a identifier string is encountered a second time, the timings are
    # combined and the count is increased
    MultiTimer("Fourth timer")
    time.sleep(0.1)
    MultiTimer("Fourth timer")
    time.sleep(0.1)

    # This could also be in a loop!
    for idx in range(0,10):
        MultiTimer("Fifth timer seen 10 times!")
        time.sleep(0.1)

    # Print the results, by calling a class method
    MultiTimer.print_timings()

    time.sleep(0.1)

    # Use reset to clear the stored data
    MultiTimer.reset()
    
    ####################################################################
    # Small performance test

    # It is smart to always run start, to assure all objects are created
    MultiTimer("Start")

    # Now to do some performance measurements
    Timer("start")
    for nr_repeats in [1,10,100,1000,1e4,1e5,1e6]:
        timer_id = "Timer " + str(nr_repeats)
        MultiTimer("loop")
        
        for idx in range(int(nr_repeats)):
            # Commend out to see the cost of the loop
            # Subtract this from the runtime with the time to get the true
            # timer cost. For me this was .04 seconds (of 1.05 seconds total)
            MultiTimer(timer_id, 3)   
            pass

        # We can use a mixture of MultiTimers and Timers
        Timer(timer_id)

    # print the results
    print ("\n Second timing results: ")
    MultiTimer.print_timings()
    print ("\n Timing of the timer: ")
    Timer.print_timings()

    #########################################################################
    # write_to_file
    file_name = "afile.cvs"
    fp = open("afile.cvs","w")
    MultiTimer.to_file_like_as_csv(fp)

    # Close and delete the file.
    fp.close()
    os.remove(file_name)

    


if __name__ == "__main__":
    run_example()