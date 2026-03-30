
# INPUT PREPARATION by subsets

def parse_input(filename, start=None, end=None):

    """ Parses input into an array of inputs and array of true results 
    
    usage: parse_input(filename, 0, 100)
        returns input_list and target_list with length 100, from lines 0 to 99 """
    
    input_list = []
    target_list = []
    for input, result in iterate_input(filename=filename, start=start, end=end):
        input_list.append(input)
        target_list.append(result)

    return input_list, target_list



def iterate_input(filename, start=None, end=None):

    """ Iterates according to the values of start and end. 
    Yields parsed input per-line as a list of input values and list of results.
        - start: inclusive
        - end: exclusive
    """

    # error checks
    if start is not None and start < 0:
        raise ValueError("start must be >= 0")
    if end is not None and end <= start:
        raise ValueError("end must be > start")
    
    input_list = []
    target_list = []

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

    """Takes a line from a file and parses it into an input and label.
    Assumes structure: 27 input values and 1 target value"""

    line_parts = line.strip().split()
    input = [float(x) for x in line_parts[:-1]]
    target = [int(line_parts[-1])]

    return input, target





# make a file of all training data
def make_training_set():
    outfilename = "data/training_set.howlin"
    with open(outfilename, 'w') as outfile:

        for i in range(1,4):
            infilename = f"data/homology_reduced_subset_{i}.howlin"
            
            with open(infilename, 'r') as infile:
                # line-by-line stream for lower RAM usage
                for line in infile:
                    outfile.write(line)



    

      







    



        
