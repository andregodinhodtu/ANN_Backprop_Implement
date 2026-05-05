import numpy as np
import sys
import random
import math
from pathlib import Path
from datetime import datetime
import time

# ------------------------------ Paths and Imports ------------------------------


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SRC = PROJECT_ROOT / "src"
SRC_NUMPY = SRC / "numpy_version"
DATA_FOLDER = PROJECT_ROOT / "data"
MODEL_FOLDER = PROJECT_ROOT / "models"

sys.path.append(str(SRC_NUMPY))
sys.path.append(str(SRC))

from data_input_numpy import parse_input
from ANN_layer_numpy import ANN_Layer_numpy
from ANN_numpy import ANN_numpy
from evaluate_numpy import report_results, evaluate

# ------------------------------------ MAIN --------------------------------------

if __name__ == "__main__":
    
    # ------------------------------ Train settings ------------------------------
    
    # Data
    TRAIN_DATA_FILE = DATA_FOLDER / "training_set.howlin"
    assert TRAIN_DATA_FILE.exists(), f"Train data file not found at: {TRAIN_DATA_FILE}"

    # Train settings 
    SEED          = 42
    TRAIN_RATIO   = 0.85
    EPOCHS        = 200
    LEARNING_RATE = 0.01
    BATCH_SIZE    = 32
    LR_DECAY      = 0.95
    DECAY_EVERY   = 20
    L2_LAMBDA     = 1e-4
    PATIENCE      = 50
    # if None auto-generate timestamped name
    MODEL_NAME    = None             
    SAVE_PATH     = str(MODEL_FOLDER)

    # Architecture
    N_LAYERS             = 4
    N_NEURONS_EACH_LAYER = [27, 32, 16, 1]
    ACTIVATION_HIDDEN    = "relu"
    ACTIVATION_OUTPUT    = "sigmoid"
    LOSS_FUNCTION        = "binarycrossentropy"

    # ------------------------------ Test settings ------------------------------
    
    # Data
    TEST_DATA_FILE = DATA_FOLDER  / "homology_reduced_subset_4.howlin"
    assert TEST_DATA_FILE.exists(), f"Test data not found at: {TEST_DATA_FILE}"
    
    # Model used to test
    MODEL_FILE = MODEL_FOLDER / "numpy_model_20260505_121015.txt"

      
    # ---------------------------- CL parsing options ---------------------------
    
    # Mode selection
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "test"):
        print("Usage: python script.py [train|test]")
        sys.exit(1)

    mode = sys.argv[1]

  
    # ------------------------------- TRAIN MODE --------------------------------

    if mode == "train":
        rng = np.random.default_rng(SEED)

        #  Data parsing
        X_all, Y_all = parse_input(str(TRAIN_DATA_FILE))
        # X_all shape: (n_samples, n_features)
        # Y_all shape: (n_samples,) or (n_samples, 1)
        
        # shape is changed to batch inside the train function
        # we admit some inconsistency 

        ones  = int(np.sum(Y_all == 1))
        zeros = len(Y_all) - ones
        print(f"Class 1: {ones}, Class 0: {zeros}, Ratio: {ones / len(Y_all):.2%}")

        # Validation split
        indices = np.arange(len(X_all))
        rng.shuffle(indices)
        split = int(TRAIN_RATIO * len(X_all))
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, Y_train = X_all[train_idx], Y_all[train_idx]
        X_val,   Y_val   = X_all[val_idx],   Y_all[val_idx]

        print(f"Train samples: {len(X_train)} ({TRAIN_RATIO:.0%}), "
              f"Val samples: {len(X_val)} ({1 - TRAIN_RATIO:.0%})")

        # Oversample minority class (0) in training set
        # Flatten Y_train for boolean masking regardless of shape (n,) or (n, 1)
        Y_flat    = Y_train.ravel()
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

        # Build the model
        ann = ANN_numpy(
            n_layers=N_LAYERS,
            n_neurons_each_layer=N_NEURONS_EACH_LAYER,
            activation_hidden=ACTIVATION_HIDDEN,
            activation_output=ACTIVATION_OUTPUT,
            loss_function=LOSS_FUNCTION,
            rng=rng,
        )

        # Train
        model_name = MODEL_NAME
        if model_name is None:
            model_name = f"numpy_model_{datetime.now():%Y%m%d_%H%M%S}.txt"

        print("Training started.")
        # measure time
        start_time = time.time()

        
        history = ann.train(
            X_train, Y_train, X_val, Y_val,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            lr_decay=LR_DECAY,
            decay_every=DECAY_EVERY,
            l2_lambda=L2_LAMBDA,
            patience=PATIENCE,
            verbose=True,
        )

        # Report and save
        report_results(ann, X_train, Y_train, X_val, Y_val, threshold=0.5)
        ann.save_model(model_name, str(TRAIN_DATA_FILE), SAVE_PATH)

        end_time = time.time()
        print(f"Training runtime: {end_time - start_time:.4f} seconds")

    # ------------------------------- TEST MODE --------------------------------
 
    elif mode == "test":
        
        # Loading the model
        ann = ANN_numpy.load_model(str(MODEL_FILE))

        X_test, Y_test = parse_input(str(TEST_DATA_FILE))

        print("Evaluation started.")
        start_time = time.time()
        
        # Evaluate handles shaping the batch 
        evaluate(ann, X_test, Y_test, name=str(TEST_DATA_FILE))

        end_time = time.time()
        print(f"Evaluation runtime: {end_time - start_time:.4f} seconds")
    

