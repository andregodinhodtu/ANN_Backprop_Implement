import random

# INPUT for stohastic backpropagation

# iterator
def load_and_split(filename, batch_size):

    """ Each iteration returns 2 lists - inputs and labels - of {batch_size} samples.
    Iteration continues until the end of file.
    
    Each call of load_and_split() will produce different batches due to random shuffling.
    """

    # load entire file
    inputs, labels = parse_input(filename)

    # zip together + shuffle randomly (no seed, so it is different for each epoch)
    data = list(zip(inputs, labels))
    random.shuffle(data)

    # split into batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        # zip the tuples into 2 tuples
        # convert them to lists
        input_batch, labels_batch = map(list, zip(*batch))
        
        yield input_batch, labels_batch



    
    



# INPUT PREPARATION by subsets

def parse_input(filename, start=None, end=None):

    """ Parses file into a list of values and list of their labels. 
    
    usage: parse_input(filename, 0, 100)
        returns input_list and label_list with length 100, from lines 0 to 99 """
    
    input_list = []
    label_list = []
    for value, label in iterate_input(filename=filename, start=start, end=end):
        input_list.append(value)
        label_list.append([label])

    return input_list, label_list



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
    Assumes structure: 27 value values and 1 label value"""

    line_parts = line.strip().split()
    if line_parts is None or len(line_parts) != 28:
        raise ValueError("Check input file - line must have 27 input values and 1 label.")
    
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



    

      







   

    



        
