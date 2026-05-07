import sys
import random
import math
from datetime import datetime
import time
import os


# ------------------------------ Paths and Imports ------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SRC = os.path.join(PROJECT_ROOT, "src")
SRC_BASE_PYTHON = os.path.join(SRC, "base_python_version")
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "models")

sys.path.append(SRC_BASE_PYTHON)
sys.path.append(SRC)

from data_input_base_python import parse_input
from ANN_layer_base_python import ANN_Layer_base_python
from ANN_base_python import ANN_base_python
from evaluate_base_python import report_results, evaluate

# ------------------------------------ MAIN --------------------------------------


if __name__ == "__main__":

    # ------------------------------ Train settings ------------------------------
    
    # Data
    TRAIN_DATA_FILE = os.path.join(DATA_FOLDER, "training_set.howlin")
    assert os.path.exists(TRAIN_DATA_FILE), f"Train data not found at: {TRAIN_DATA_FILE}"

    # Train settings 
    SEED          = 42
    TRAIN_RATIO   = 0.85
    EPOCHS        = 3
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
    TEST_DATA_FILE = os.path.join(DATA_FOLDER, "homology_reduced_subset_4.howlin")
    assert os.path.exists(TEST_DATA_FILE), f"Test data not found at: {TEST_DATA_FILE}"
    
    # Model used to test
    MODEL_FILE = os.path.join(MODEL_FOLDER, "base_python_model_20260505_121602.txt")

      
    # ---------------------------- CL parsing options ---------------------------
    
    # Mode selection
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "test"):
        print("Usage: python script.py [train|test]")
        sys.exit(1)

    mode = sys.argv[1]

  
    # ------------------------------- TRAIN MODE --------------------------------
   
    if mode == "train":
        rng = random.Random(SEED)

        # Data parsing
        X_all, Y_all = parse_input(str(TRAIN_DATA_FILE))

        ones = sum(1 for y in Y_all if y[0][0] == 1)
        print(ones)
        zeros = len(Y_all) - ones
        print(f"Class 1: {ones}, Class 0: {zeros}, Ratio: {ones/len(Y_all):.2%}")

        # Validation split 
        indices = list(range(len(X_all)))
        rng.shuffle(indices)
        split = int(TRAIN_RATIO * len(X_all))
        X_train = [X_all[i] for i in indices[:split]]
        Y_train = [Y_all[i] for i in indices[:split]]
        X_val   = [X_all[i] for i in indices[split:]]
        Y_val   = [Y_all[i] for i in indices[split:]]
        print(f"Train samples: {len(X_train)} ({TRAIN_RATIO:.0%}), "
              f"Val samples: {len(X_val)} ({1-TRAIN_RATIO:.0%})")

        # Oversample minority class (0) in training set 
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

        # Build the model
        ann = ANN_base_python(
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
            model_name = f"base_python_model_{datetime.now():%Y%m%d_%H%M%S}.txt"

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
        )

        end_time = time.time()
        print(f"Training runtime: {end_time - start_time:.4f} seconds")

        # Report and save
        report_results(ann, X_train, Y_train, X_val, Y_val, threshold=0.5)
        ann.save_model(model_name, str(TRAIN_DATA_FILE), SAVE_PATH)

   
    # ------------------------------- TEST MODE --------------------------------

    elif mode == "test":
        
        # Loading the model
        ann = ANN_base_python.load_model(MODEL_FILE)

        X_test, Y_test = parse_input(str(TEST_DATA_FILE))

        print("Evaluation started.")
        start_time = time.time()
        
        # Evaluate handles shaping the batch 
        evaluate(ann, X_test, Y_test, name=str(TEST_DATA_FILE))

        end_time = time.time()
        print(f"Evaluation runtime: {end_time - start_time:.4f} seconds")
    
    