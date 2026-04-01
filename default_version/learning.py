import ANN
import data_prep


filename = "../data/training_set.howlin"

# 1 SAMPLE CHECK

#get 1 sample
values, label = data_prep.parse_input(filename, 0, 64)

print(f"The length of the input is: {len(values[0])}")

print(f"\n" + "#"*74)       
print(f"#"*30 + " Start Script " + "#"*30)        
print(f"#"*74 + "\n")

#print(values)
print(label)

label = [[x] for x in label]

n_neurons_each_layer = [len(values[0]),16, 8, 1]
n_layers = len(n_neurons_each_layer)

neural_network = ANN.ANN(n_layers,
                     n_neurons_each_layer,
                     activation_hidden="relu",
                     activation_output="sigmoid",
                     loss_function="BinaryCrossEntropy")
                     
#neural_network.backprop_one_training_example(values, label)

neural_network.backpropation_batch(values, label, steps = 1000, learning_rate = 0.1)

final_predict = [neural_network.prediction([x]) for x in values]

for val,lab in zip(final_predict,label):
    print(f"This is the value predicted {val} and the label {lab}")
    if round(val[0][0]) != lab[0][0]:
        print("ERROR")





