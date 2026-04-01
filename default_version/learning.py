import ANN
import data_prep


filename = "../data/training_set.howlin"

# 1 SAMPLE CHECK

#get 1 sample
values, label = data_prep.parse_input(filename, 0, 1)

print(f"The length of the input is: {len(values[0])}")

print(f"\n" + "#"*74)       
print(f"#"*30 + " Start Script " + "#"*30)        
print(f"#"*74 + "\n")

n_neurons_each_layer = [len(values[0]),21,18,15,9,6,3,1]
n_layers = len(n_neurons_each_layer)

neural_network = ANN.ANN(n_layers,
                     n_neurons_each_layer,
                     "sigmoid",
                     "BinaryCrossEntropy")
                     
neural_network.backprop_one_training_example(values, label)





