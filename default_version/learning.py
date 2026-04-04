import ANN
import data_prep


filename = "data/training_set.howlin"

n_neurons_each_layer = [27, 16, 16, 8, 1]
n_layers = len(n_neurons_each_layer)

neural_network = ANN.ANN(n_layers,
                        n_neurons_each_layer,
                        activation_hidden="relu",
                        activation_output="sigmoid",
                        loss_function="BinaryCrossEntropy")

# loop through epoches
num_epochs = 20

for epoch in range(num_epochs):
  #  print(f"Epoch {epoch}")
    # loop through batches (1 epoch)
    count = 0
    for input_batch, label_batch in data_prep.load_and_split(filename, 128):
        #print(f"Epoch {epoch}, Batch {count}")
        count += 1

        # steps=1 to compute gradient only once for each batch
        
        neural_network.backpropagation_batch(input_batch, label_batch, steps=1, learning_rate=0.01, verbose=False)
        final_predict = [neural_network.prediction([x]) for x in input_batch]

        err = 0
        n = 0
        for val,lab in zip(final_predict, label_batch):
            n += 1
            #print(f"Value predicted: {val[0][0]:.2f} vs true: {lab[0][0]}")
            if round(val[0][0]) != lab[0][0]:
                err += 1
               # print("ERROR")
       # print(f"Wrong {err} out of {n}: {err/n:.3f}")


    # validation after each epoch:
    val_inputs, val_labels = data_prep.parse_input("data/homology_reduced_subset_4.howlin")
    val_preds = [neural_network.prediction([x]) for x in val_inputs]

    val_err = 0
    for val, lab in zip(val_preds, val_labels):
        if round(val[0][0]) != lab[0][0]:
            val_err += 1
    print(f"Validation error after epoch {epoch}: {val_err/len(val_labels):.3f}")
            








# test learning with backpropagation for 1 batch 
def one_batch_ann_check():
    #get 1 sample
    values, label = data_prep.parse_input(filename, 0, 64)

    print(f"The length of the input is: {len(values[0])}")

    print(f"\n" + "#"*74)       
    print(f"#"*30 + " Start Script " + "#"*30)        
    print(f"#"*74 + "\n")

    #print(values)
    print(label)

    n_neurons_each_layer = [len(values[0]),16, 8, 1]
    n_layers = len(n_neurons_each_layer)

    neural_network = ANN.ANN(n_layers,
                        n_neurons_each_layer,
                        activation_hidden="relu",
                        activation_output="sigmoid",
                        loss_function="BinaryCrossEntropy")
                        
    #neural_network.backprop_one_training_example(values, label)

    neural_network.backpropagation_batch(values, label, steps = 1000, learning_rate = 0.1)

    final_predict = [neural_network.prediction([x]) for x in values]

    for val,lab in zip(final_predict,label):
        print(f"This is the value predicted {val} and the label {lab}")
        if round(val[0][0]) != lab[0][0]:
            print("ERROR")

#one_batch_ann_check()
