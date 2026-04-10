import data_prep
import random
from ANN_layer import ANN_Layer
from ANN import ANN

def train_real_data():
    import math
    X, Y = data_prep.parse_input("../data/training_set.howlin")

    # Check output distribution of untrained network
    ann = ANN(
        n_layers=4,
        n_neurons_each_layer=[27, 64, 32, 1],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="BinaryCrossEntropy"
    )

    # He/Xavier init (same as before)
    for layer in ann.layers:
        is_output = (layer == ann.layers[-1])
        if is_output:
            limit = math.sqrt(6 / (layer.n_neurons_input + layer.n_neurons_output))
            layer.weights = [
                [random.uniform(-limit, limit) for _ in range(layer.n_neurons_input)]
                for _ in range(layer.n_neurons_output)
            ]
            layer.biases = [[0.0] for _ in range(layer.n_neurons_output)]
        else:
            std = math.sqrt(2 / layer.n_neurons_input)
            layer.weights = [
                [random.gauss(0, std) for _ in range(layer.n_neurons_input)]
                for _ in range(layer.n_neurons_output)
            ]
            layer.biases = [[0.01] for _ in range(layer.n_neurons_output)]

    # === DIAGNOSTIC: check predictions before training ===
    preds_before = [ann.prediction(x)[0][0] for x in X[:20]]
    print("Sample predictions BEFORE training:")
    print([round(p, 3) for p in preds_before])
    print(f"Prediction mean: {sum(preds_before)/len(preds_before):.3f}")
    print(f"Prediction min:  {min(preds_before):.3f}")
    print(f"Prediction max:  {max(preds_before):.3f}")

    print("\n--- TRAINING START ---\n")

    ann.train(
        X, Y,
        epochs=400,          # more epochs since it's still descending
        learning_rate=0.1,
        batch_size=32,
        verbose=True,
        lr_decay=0.95,       # multiply lr by 0.95 every 20 epochs
        decay_every=20
    )

    # === DIAGNOSTIC: check predictions after training ===
    preds_after = [ann.prediction(x)[0][0] for x in X[:20]]
    labels = [y[0][0] for y in Y[:20]]
    print("\nSample predictions AFTER training:")
    for p, l in zip(preds_after, labels):
        print(f"  pred: {p:.3f}  label: {l}")

    # Accuracy on full training set
    correct = 0
    for x, y in zip(X, Y):
        pred = ann.prediction(x)[0][0]
        predicted_class = 1 if pred >= 0.5 else 0
        if predicted_class == y[0][0]:
            correct += 1
    print(f"\nTraining accuracy: {correct}/{len(X)} = {correct/len(X):.2%}")
        
if __name__ == "__main__":

    train_real_data()
    