import data_prep
import random
from ANN_layer_base_python import ANN_Layer_base_python
from ANN_base_python import ANN_base_python
import math

def train_real_data():

    X_all, Y_all = data_prep.parse_input("../../data/training_set.howlin")
    
    ones = sum(1 for y in Y_all if y[0][0] == 1)
    zeros = len(Y_all) - ones
    print(f"Class 1: {ones}, Class 0: {zeros}, Ratio: {ones/len(Y_all):.2%}")

    # === VALIDATION SPLIT (85% train, 15% val) ===
    indices = list(range(len(X_all)))
    random.shuffle(indices)
    split = int(0.85 * len(X_all))

    X_train = [X_all[i] for i in indices[:split]]
    Y_train = [Y_all[i] for i in indices[:split]]
    X_val   = [X_all[i] for i in indices[split:]]
    Y_val   = [Y_all[i] for i in indices[split:]]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # === OVERSAMPLE MINORITY CLASS (0) IN TRAINING SET ===
    ones_idx  = [i for i, y in enumerate(Y_train) if y[0][0] == 1]
    zeros_idx = [i for i, y in enumerate(Y_train) if y[0][0] == 0]

    diff = len(ones_idx) - len(zeros_idx)
    if diff > 0:
        extra = random.choices(zeros_idx, k=diff)
        X_train += [X_train[i] for i in extra]
        Y_train += [Y_train[i] for i in extra]
        combined = list(zip(X_train, Y_train))
        random.shuffle(combined)
        X_train, Y_train = list(zip(*combined))
        X_train, Y_train = list(X_train), list(Y_train)

    ones_after  = sum(1 for y in Y_train if y[0][0] == 1)
    zeros_after = sum(1 for y in Y_train if y[0][0] == 0)
    print(f"After oversampling — Class 1: {ones_after}, Class 0: {zeros_after}")

    ann = ANN_base_python(
        n_layers=4,
        n_neurons_each_layer=[27, 32, 16, 1],
        activation_hidden="leaky_relu",
        activation_output="sigmoid",
        loss_function="BinaryCrossEntropy"
    )

    # He/Xavier init
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

    print("\n--- TRAINING START ---\n")

    # === EARLY STOPPING SETUP ===
    best_val_loss = float('inf')
    best_weights = None
    patience = 50
    epochs_no_improve = 0
    best_epoch = 0

    def compute_val_loss(X, Y):
        total_loss = 0.0
        for x, y in zip(X, Y):
            pred = ann.prediction(x)[0][0]
            label = y[0][0]
            pred = max(min(pred, 1 - 1e-7), 1e-7)
            total_loss += -(label * math.log(pred) + (1 - label) * math.log(1 - pred))
        return total_loss / len(X)

    def save_weights():
        return [(
            [row[:] for row in layer.weights],
            [row[:] for row in layer.biases]
        ) for layer in ann.layers]

    def restore_weights(saved):
        for layer, (w, b) in zip(ann.layers, saved):
            layer.weights = [row[:] for row in w]
            layer.biases  = [row[:] for row in b]

    # Train one epoch at a time so we can check val loss
    current_lr = 0.01
    epochs = 201
    for epoch in range(1, epochs):

        # LR decay every 20 epochs
        if epoch > 1 and (epoch - 1) % 20 == 0:
            current_lr *= 0.95
            print(f"  [LR decayed to {current_lr:.6f}]")

        ann.train(
            X_train, Y_train,
            epochs=1,
            learning_rate=current_lr,
            batch_size=32,
            verbose=False,
            lr_decay=1.0,
            decay_every=9999,
            l2_lambda = 1e-4
        )

        val_loss = compute_val_loss(X_val, Y_val)
        train_loss = compute_val_loss(X_train, Y_train)
        print(f"Epoch {epoch}/{epochs-1} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_weights = save_weights()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n*** Early stopping at epoch {epoch} (best was epoch {best_epoch}) ***")
                break

    print(f"\nRestoring best weights from epoch {best_epoch} (val loss: {best_val_loss:.6f})")
    restore_weights(best_weights)

    # === SAMPLE PREDICTIONS ===
    preds_after = [ann.prediction(x)[0][0] for x in X_train[:20]]
    labels = [y[0][0] for y in Y_train[:20]]
    print("\nSample predictions AFTER training:")
    for p, l in zip(preds_after, labels):
        print(f"  pred: {p:.3f}  label: {l}")

    # === TRAINING ACCURACY ===
    correct = sum(
        1 for x, y in zip(X_train, Y_train)
        if (1 if ann.prediction(x)[0][0] >= 0.5 else 0) == y[0][0]
    )
    print(f"\nTraining accuracy: {correct}/{len(X_train)} = {correct/len(X_train):.2%}")

    # === VALIDATION ACCURACY ===
    correct = sum(
        1 for x, y in zip(X_val, Y_val)
        if (1 if ann.prediction(x)[0][0] >= 0.5 else 0) == y[0][0]
    )
    print(f"Validation accuracy: {correct}/{len(X_val)} = {correct/len(X_val):.2%}")

    # === TEST ACCURACY ===
    X_test, Y_test = data_prep.parse_input("../../data/homology_reduced_subset_4.howlin")
    correct = sum(
        1 for x, y in zip(X_test, Y_test)
        if (1 if ann.prediction(x)[0][0] >= 0.5 else 0) == y[0][0]
    )
    print(f"Testing accuracy: {correct}/{len(X_test)} = {correct/len(X_test):.2%}")

if __name__ == "__main__":
    train_real_data()
    