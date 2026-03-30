
# make a file of all training data

outfilename = "data/training_set.howlin"
with open(outfilename, 'w') as outfile:

    for i in range(1,4):
        infilename = f"data/homology_reduced_subset_{i}.howlin"
        
        with open(infilename, 'r') as infile:
            # line-by-line stream for lower RAM usage
            for line in infile:
                outfile.write(line)





        
