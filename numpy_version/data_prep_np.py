import numpy as np

# INPUT PREPARATION by subsets

def parse_input(filename, start=None, end=None):

    """ Parses file into a NumPy arrays. 
    
    Usage: parse_input(filename, 0, 100)
        returns input_list and label_list with length 100, from lines 0 to 99 
        
    Returns: X = (n_samples, 27) float array of input values
            y = (1, n_samples) int array of labels"""
    
    input_list = []
    label_list = []

    for value, label in iterate_input(filename=filename, start=start, end=end):
        input_list.append(value)
        label_list.append(label)

    X = np.array(input_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int32)

    return X, y



def iterate_input(filename, start=None, end=None):

    """ Iterates according to start and end. 
    Yields parsed input per-line as a list of values and list of results.
        - start: inclusive
        - end: exclusive
    """

    # error checks
    if start is not None and start < 0:
        raise ValueError("start must be >= 0")
    if end is not None and end < 0:
        raise ValueError("end must be >= 0")
    if start is not None and end is not None and end < start:
        raise ValueError("end must be >= start")

    # borders
    lo = 0 if start is None else start
    hi = float("inf") if end is None else end # inf -> end is undefined

    with open(filename, 'r') as infile:  
        for i, line in enumerate(infile):
            if i < lo:
                continue
            if i >= hi:
                break

            parsed = parse_line(line)
            yield parsed 



def parse_line(line):

    """Takes a line from a file and parses it into an value and label.
    Assumes structure: 27 feature values and 1 label value"""

    line_parts = line.strip().split()
    if line_parts is None or len(line) != 28: # 27 features + 1 label
        raise ValueError(f"Check input file - expected 28, got {len(line_parts)}")
    
    
    value = [float(x) for x in line_parts[:-1]]
    label = [int(line_parts[-1])]

    return value, label





# make a file of all training data (discarded, one-use)

def make_training_set():
    outfilename = "data/training_set.howlin"
    with open(outfilename, 'w') as outfile:

        for i in range(1,4):
            infilename = f"data/homology_reduced_subset_{i}.howlin"
            
            with open(infilename, 'r') as infile:
                # line-by-line stream for lower RAM usage
                for line in infile:
                    outfile.write(line)



    

      







    



        
