import numpy as np
import sys
import random
import math
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SRC = PROJECT_ROOT / "src" 
SRC_NUMPY = SRC / "numpy_version"
TRAIN_DATA_FILE = PROJECT_ROOT / "data" / "training_set.howlin"
TEST_DATA_FILE = PROJECT_ROOT / "data" / "homology_reduced_subset_4.howlin"
MODEL_FOLDER = PROJECT_ROOT / "models"

sys.path.append(str(SRC_NUMPY))
sys.path.append(str(SRC))

from data_input_numpy import parse_input
from ANN_layer_numpy import ANN_Layer_numpy
from ANN_numpy import ANN_numpy
from evaluate import report_results, evaluate

# Sanity check — fails fast with a clear message if the path is wrong
assert TRAIN_DATA_FILE.exists(), f"Train data file not found at: {TRAIN_DATA_FILE}"
assert TEST_DATA_FILE.exists(), f"Test data not found at: {TEST_DATA_FILE}"

def train_real_data():


    # helpers
    def compute_loss(X, Y):
        preds = np.array([ann.prediction(x) for x in X]).reshape(len(X), -1)
        labels = Y.reshape(len(Y), -1)
        loss_func = ANN.LOSS_FUNCTIONS["binary_cross_entropy"]["func"]
        return float(loss_func(labels, preds))

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

    X_test_raw, y_test_raw = parse_input(
        "data/homology_reduced_subset_4.howlin")
    X_test = X_test_raw.reshape(-1, 27, 1).astype(np.float32)
    Y_test = y_test_raw.reshape(-1, 1, 1).astype(np.float32)

    test_acc = compute_accuracy(X_test, Y_test)
    print(f"Test       accuracy: {test_acc:.2%}   ({int(test_acc*len(X_test))}/{len(X_test)})")

def train_data_handling(seed=42, train_ratio=0.85):
    rng = np.random.default_rng(seed)

    # Data parsing
    X_all, Y_all = parse_input(str(TRAIN_DATA_FILE))
    # X_all shape: (n_samples, n_features)
    # Y_all shape: (n_samples,) or (n_samples, 1)

    print(X_all)
    ones  = int(np.sum(Y_all == 1))
    zeros = len(Y_all) - ones
    print(f"Class 1: {ones}, Class 0: {zeros}, Ratio: {ones / len(Y_all):.2%}")

    # === VALIDATION SPLIT ===
    indices = np.arange(len(X_all))
    rng.shuffle(indices)
    split = int(train_ratio * len(X_all))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val,   Y_val   = X_all[val_idx],   Y_all[val_idx]

    print(f"Train samples: {len(X_train)} ({train_ratio:.0%}), "
          f"Val samples: {len(X_val)} ({1 - train_ratio:.0%})")

    # === OVERSAMPLE MINORITY CLASS (0) IN TRAINING SET ===
    # Flatten Y_train for boolean masking regardless of shape (n,) or (n, 1)
    Y_flat   = Y_train.ravel()
    ones_idx  = np.where(Y_flat == 1)[0]
    zeros_idx = np.where(Y_flat == 0)[0]

    diff = len(ones_idx) - len(zeros_idx)
    if diff > 0:
        extra = rng.choice(zeros_idx, size=diff, replace=True)
        X_train = np.concatenate([X_train, X_train[extra]], axis=0)
        Y_train = np.concatenate([Y_train, Y_train[extra]], axis=0)

        # Shuffle train set so the duplicates aren't all at the end
        perm = rng.permutation(len(X_train))
        X_train, Y_train = X_train[perm], Y_train[perm]

    ones_after  = int(np.sum(Y_train == 1))
    zeros_after = int(np.sum(Y_train == 0))
    print(f"After oversampling — Class 1: {ones_after}, Class 0: {zeros_after}")

    return X_train, Y_train, X_val, Y_val, rng
     
def test_data_handling():
    # Data parsing
    X_test, Y_test = parse_input(str(TEST_DATA_FILE))
    return X_test, Y_test
    
def create_model(n_layers=4,
                 n_neurons_each_layer=None,
                 activation_hidden="leaky_relu",
                 activation_output="sigmoid",
                 loss_function="BinaryCrossEntropy"):
    """
    Build and return an untrained ANN with the chosen architecture.
    """
    ann = ANN_numpy(
        n_layers=n_layers,
        n_neurons_each_layer=n_neurons_each_layer,
        activation_hidden=activation_hidden,
        activation_output=activation_output,
        loss_function=loss_function,
    )
    return ann

def train_model(ann, X_train, Y_train, X_val, Y_val, rng,
                data_name = None,
                model_name=None,
                save_path="../models",
                epochs=200,
                learning_rate=0.01,
                batch_size=32,
                lr_decay=0.95,
                decay_every=20,
                l2_lambda=1e-4,
                patience=50):
    """
    Train an ANN, report results, and save the model.
    
    Most of the heavy lifting (training loop, early stopping, weight snapshots)
    happens inside ann.train(). This function just orchestrates and reports.
    """
    
    # Auto-generate a timestamped name if none provided
    if model_name is None:
        model_name = f"base_python_model_{datetime.now():%Y%m%d_%H%M%S}.txt"
    
    # === TRAIN ===
    history = ann.train(
        X_train, Y_train, X_val, Y_val,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lr_decay=lr_decay,
        decay_every=decay_every,
        l2_lambda=l2_lambda,
        patience=patience,
        rng=rng,
    )
    
    # === REPORT RESULTS ===
    report_results(ann, X_train, Y_train, X_val, Y_val, threshold=0.5)
    
    # === SAVE MODEL ===
    ann.save_model(model_name, data_name, save_path)
    
    return history

def test_model():
    
    # === TEST ACCURACY ===
    X_test, Y_test = parse_input("../../data/homology_reduced_subset_4.howlin")
    correct = sum(
        1 for x, y in zip(X_test, Y_test)
        if (1 if ann.prediction(x)[0][0] >= 0.5 else 0) == y[0][0]
    )
    print(f"Testing accuracy: {correct}/{len(X_test)} = {correct/len(X_test):.2%}")
    
    pass

if __name__ == "__main__":
    
    # Data Handling train and validation sets
    X_train, Y_train, X_val, Y_val, rng = train_data_handling(seed=42, train_ratio = 0.85)
    
    
    # Create Neural Network
    ann = create_model(n_layers=4,
                       n_neurons_each_layer=[27, 32, 16, 1],
                       activation_hidden="relu",
                       activation_output="sigmoid",
                       loss_function="BinaryCrossEntropy")
    
    print(ann.n_layers)
    
    """
    # Train the model
    train_model(ann, X_train, Y_train, X_val, Y_val, rng,
                data_name = str(TRAIN_DATA_FILE),
                model_name=None,
                save_path="../models",
                epochs=5,
                learning_rate=0.01,
                batch_size=32,
                lr_decay=0.95,
                decay_every=20,
                l2_lambda=1e-4,
                patience=50)
    
    # test_model()
    ann_test = ANN_base_python.load_model(MODEL_FOLDER / "model_20260428_183826.txt")
    
    X_test, Y_test = test_data_handling()
    
    evaluate(ann_test, X_test, Y_test, name=str(TEST_DATA_FILE))"""
    
    
    

