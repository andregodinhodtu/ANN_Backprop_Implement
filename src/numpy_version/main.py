import numpy as np
import data_prep_np
from ANN import ANN

import numpy as np
import data_prep_np
from ANN import ANN


def train_real_data():

    # Load data 
    X_all_raw, y_all_raw = data_prep_np.parse_input("data/training_set.howlin")

    # Reshape for ANN: (n_samples, n_features, 1) and (n_samples, 1, 1)
    X_all = X_all_raw.reshape(-1, 27, 1).astype(np.float32)
    Y_all = y_all_raw.reshape(-1, 1, 1).astype(np.float32)

    ones  = int(Y_all.sum())
    zeros = len(Y_all) - ones
    print(f"Class 1: {ones}, Class 0: {zeros}, Ratio: {ones/len(Y_all):.2%}")

    # validation split (85 / 15) 
    indices = np.random.permutation(len(X_all))
    split   = int(0.85 * len(X_all))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val,   Y_val   = X_all[val_idx],   Y_all[val_idx]
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # oversample minority class in training set
    ones_mask  = (Y_train[:, 0, 0] == 1)
    zeros_mask = ~ones_mask
    n_ones, n_zeros = ones_mask.sum(), zeros_mask.sum()

    if n_ones > n_zeros:
        diff      = n_ones - n_zeros
        dup_idx   = np.random.choice(np.where(zeros_mask)[0], size=diff, replace=True)
        X_train   = np.concatenate([X_train, X_train[dup_idx]], axis=0)
        Y_train   = np.concatenate([Y_train, Y_train[dup_idx]], axis=0)
        shuffle   = np.random.permutation(len(X_train))
        X_train, Y_train = X_train[shuffle], Y_train[shuffle]

    ones_after  = int(Y_train[:, 0, 0].sum())
    zeros_after = len(Y_train) - ones_after
    print(f"After oversampling — Class 1: {ones_after}, Class 0: {zeros_after}")

    # build ANN
    ann = ANN(
        n_layers=4,
        n_neurons_each_layer=[27, 32, 16, 1],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binary_cross_entropy",
    )



    # helpers
    def compute_loss(X, Y):
        preds = np.array([ann.prediction(x) for x in X]).reshape(len(X), -1)
        labels = Y.reshape(len(Y), -1)
        return float(ANN.binary_cross_entropy(labels, preds))

    def compute_accuracy(X, Y):
        preds  = np.array([ann.prediction(x) for x in X]).reshape(len(X))
        labels = Y.reshape(len(Y))
        return np.mean((preds >= 0.5).astype(int) == labels.astype(int))

    def save_weights():
        return [(layer.weights.copy(), layer.biases.copy()) for layer in ann.layers]

    def restore_weights(saved):
        for layer, (w, b) in zip(ann.layers, saved):
            layer.weights = w.copy()
            layer.biases  = b.copy()

    # training loop - early stopping
    print("\n--- TRAINING START ---\n")

    best_val_loss      = float("inf")
    best_weights       = None
    best_epoch         = 0
    patience           = 50
    epochs_no_improve  = 0
    current_lr         = 0.01
    total_epochs       = 201

    for epoch in range(1, total_epochs):

        # LR decay
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
            l2_lambda=1e-4,
        )

        train_loss = compute_loss(X_train, Y_train)
        val_loss   = compute_loss(X_val,   Y_val)
        print(f"Epoch {epoch}/{total_epochs-1} — "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_epoch        = epoch
            epochs_no_improve = 0
            best_weights      = save_weights()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n*** Early stopping at epoch {epoch} "
                      f"(best was epoch {best_epoch}) ***")
                break

    print(f"\nRestoring best weights from epoch {best_epoch} "
          f"(val loss: {best_val_loss:.6f})")
    restore_weights(best_weights)

    # sample predictions
    print("\nSample predictions AFTER training (first 20 training samples):")
    for x, y in zip(X_train[:20], Y_train[:20]):
        prob  = ann.prediction(x).item()
        label = int(y.flat[0])
        print(f"  pred: {prob:.3f}  label: {label}")

    # accuracy ?
    train_acc = compute_accuracy(X_train, Y_train)
    val_acc   = compute_accuracy(X_val,   Y_val)
    print(f"\nTraining   accuracy: {train_acc:.2%} ({int(train_acc*len(X_train))}/{len(X_train)})")
    print(f"Validation accuracy: {val_acc:.2%}   ({int(val_acc*len(X_val))}/{len(X_val)})")

    X_test_raw, y_test_raw = data_prep_np.parse_input(
        "data/homology_reduced_subset_4.howlin")
    X_test = X_test_raw.reshape(-1, 27, 1).astype(np.float32)
    Y_test = y_test_raw.reshape(-1, 1, 1).astype(np.float32)

    test_acc = compute_accuracy(X_test, Y_test)
    print(f"Test       accuracy: {test_acc:.2%}   ({int(test_acc*len(X_test))}/{len(X_test)})")


if __name__ == "__main__":
    train_real_data()