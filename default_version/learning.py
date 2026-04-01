# import ANN
import data_prep


filename = "data/training_set.howlin"

# 1 SAMPLE CHECK

#get 1 sample
values, label = data_prep.parse_input(filename, 0, 1)


