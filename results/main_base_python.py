import sys
import random
import math
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SRC = PROJECT_ROOT / "src" 
SRC_BASE_PYTHON = SRC / "base_python_version"
TRAIN_DATA_FILE = PROJECT_ROOT / "data" / "training_set.howlin"
TEST_DATA_FILE = PROJECT_ROOT / "data" / "homology_reduced_subset_4.howlin"
MODEL_FOLDER = PROJECT_ROOT / "models"


sys.path.append(str(SRC_BASE_PYTHON))
sys.path.append(str(SRC))

from data_input_base_python import parse_input
from ANN_layer_base_python import ANN_Layer_base_python
from ANN_base_python import ANN_base_python
from evaluate_base_python import report_results, evaluate

# Sanity check — fails fast with a clear message if the path is wrong
assert TRAIN_DATA_FILE.exists(), f"Train data file not found at: {TRAIN_DATA_FILE}"
assert TEST_DATA_FILE.exists(), f"Test data not found at: {TEST_DATA_FILE}"

def train_data_handling(seed=42, train_ratio=0.85):
    rng = random.Random(seed)

    # Data parsing
    X_all, Y_all = parse_input(str(TRAIN_DATA_FILE))

    ones = sum(1 for y in Y_all if y[0][0] == 1)
    zeros = len(Y_all) - ones
    print(f"Class 1: {ones}, Class 0: {zeros}, Ratio: {ones/len(Y_all):.2%}")

    # === VALIDATION SPLIT ===
    indices = list(range(len(X_all)))
    rng.shuffle(indices)
    split = int(train_ratio * len(X_all))
    X_train = [X_all[i] for i in indices[:split]]
    Y_train = [Y_all[i] for i in indices[:split]]
    X_val   = [X_all[i] for i in indices[split:]]
    Y_val   = [Y_all[i] for i in indices[split:]]
    print(f"Train samples: {len(X_train)} ({train_ratio:.0%}), "
          f"Val samples: {len(X_val)} ({1-train_ratio:.0%})")

    # === OVERSAMPLE MINORITY CLASS (0) IN TRAINING SET ===
    ones_idx  = [i for i, y in enumerate(Y_train) if y[0][0] == 1]
    zeros_idx = [i for i, y in enumerate(Y_train) if y[0][0] == 0]
    diff = len(ones_idx) - len(zeros_idx)
    if diff > 0:
        extra = rng.choices(zeros_idx, k=diff)
        X_train += [X_train[i] for i in extra]
        Y_train += [Y_train[i] for i in extra]
        combined = list(zip(X_train, Y_train))
        rng.shuffle(combined)
        X_train, Y_train = list(zip(*combined))
        X_train, Y_train = list(X_train), list(Y_train)

    ones_after  = sum(1 for y in Y_train if y[0][0] == 1)
    zeros_after = sum(1 for y in Y_train if y[0][0] == 0)
    print(f"After oversampling — Class 1: {ones_after}, Class 0: {zeros_after}")

    return X_train, Y_train, X_val, Y_val, rng
     
def test_data_handling():
    # Data parsing
    X_test, Y_test = parse_input(str(TEST_DATA_FILE))
    return X_test, Y_test
    
def create_model(n_layers,
                 n_neurons_each_layer,
                 activation_hidden,
                 activation_output,
                 loss_function,
                 rng):
    """
    Build and return an untrained ANN with the chosen architecture.
    """
    ann = ANN_base_python(
        n_layers=n_layers,
        n_neurons_each_layer=n_neurons_each_layer,
        activation_hidden=activation_hidden,
        activation_output=activation_output,
        loss_function=loss_function,
        rng = rng
    )
    return ann

def train_model(ann, X_train, Y_train, X_val, Y_val,
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
                       loss_function="binarycrossentropy",
                       rng = rng)
    
    
    # Train the model
    train_model(ann, X_train, Y_train, X_val, Y_val,
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
    #ann_test = ANN_base_python.load_model(MODEL_FOLDER / "model_20260428_183826.txt")
    
    #X_test, Y_test = test_data_handling()
    
    #evaluate(ann_test, X_test, Y_test, name=str(TEST_DATA_FILE))"""
    
    
    