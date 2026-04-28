"""
evaluate.py
-----------
Evaluation utilities for a trained binary-classification ANN.

Quick reference of what's here:

  Basic metrics (single numbers)
    accuracy                - fraction of correct predictions
    confusion_matrix        - dict of TN/FP/FN/TP counts
    classification_metrics  - dict with precision, recall, F1, etc.

  Printing helpers
    print_confusion_matrix
    print_sample_predictions

  High-level reports (what you'll usually call)
    report_results          - quick post-training summary on train + val
    evaluate                - full report on one dataset

All functions take a trained ANN object that exposes:
    ann.prediction(x)       -> [[float]]   (column vector, single output)
    ann.compute_loss(X, Y)  -> float       (mean loss over a dataset)

Inputs X, Y are lists of column vectors:
    X = [[[v1], [v2], ..., [vN]], ...]
    Y = [[[label]], ...]                   # label is 0 or 1
"""

def accuracy(ann, X, Y, threshold=0.5):
    """Binary classification accuracy at the given decision threshold."""
    correct = sum(
        1 for x, y in zip(X, Y)
        if (1 if ann.prediction(x)[0][0] >= threshold else 0) == y[0][0]
    )
    return correct / len(X)

def confusion_matrix(ann, X, Y, threshold=0.5):
    """
    Compute the four counts of the binary confusion matrix.

    Returns:
    --------
    dict with keys: 'tn', 'fp', 'fn', 'tp'
        tn (true negative):  predicted 0, actual 0
        fp (false positive): predicted 1, actual 0
        fn (false negative): predicted 0, actual 1
        tp (true positive):  predicted 1, actual 1
    """
    tn = fp = fn = tp = 0
    for x, y in zip(X, Y):
        pred = 1 if ann.prediction(x)[0][0] >= threshold else 0
        label = int(y[0][0])
        if   label == 0 and pred == 0: tn += 1
        elif label == 0 and pred == 1: fp += 1
        elif label == 1 and pred == 0: fn += 1
        elif label == 1 and pred == 1: tp += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

def classification_metrics(ann, X, Y, threshold=0.5):
    """
    Compute a full set of classification metrics.

    Returns:
    --------
    dict with keys:
        accuracy    - overall fraction correct
        precision   - of predicted positives, how many were correct
        recall      - of actual positives, how many we caught (sensitivity)
        specificity - of actual negatives, how many we correctly rejected
        f1          - harmonic mean of precision and recall
    """
    cm = confusion_matrix(ann, X, Y, threshold)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = tp + tn + fp + fn

    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "accuracy":    acc,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "f1":          f1,
    }

def print_confusion_matrix(ann, X, Y, threshold=0.5):
    """Print a nicely formatted confusion matrix."""
    cm = confusion_matrix(ann, X, Y, threshold)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    print(f"\nConfusion matrix (threshold={threshold}):")
    print(f"                  Pred 0    Pred 1")
    print(f"   Actual 0   {tn:>8}  {fp:>8}")
    print(f"   Actual 1   {fn:>8}  {tp:>8}")

def report_results(ann, X_train, Y_train, X_val, Y_val, threshold=0.5):
    """
    Quick post-training summary: sample predictions plus train and val accuracy.

    This is the lightweight report. For a full breakdown (precision, recall,
    confusion matrix, etc.), use `evaluate` instead.
    """

    train_acc = accuracy(ann, X_train, Y_train, threshold=threshold)
    val_acc   = accuracy(ann, X_val,   Y_val,   threshold=threshold)

    print(f"\nTraining accuracy:   {train_acc:.2%}")
    print(f"Validation accuracy: {val_acc:.2%}")

def evaluate(ann, X, Y, threshold = 0.5, name="Dataset"):
    """
    Run a full evaluation report on a single dataset.

    Prints loss, headline metrics, and confusion matrix.

    Parameters:
    -----------
    ann          : trained ANN
    X, Y         : data and labels
    threshold    : decision threshold for converting probability -> class
    name         : label printed at the top of the report
    show_samples : if True, also print 20 sample predictions

    Returns:
    --------
    dict with all metrics plus 'loss'
    """
    loss = ann.compute_loss(X, Y)
    metrics = classification_metrics(ann, X, Y, threshold)
    metrics["loss"] = loss

    print(f"\n=== {name} Evaluation (threshold={threshold}) ===")
    print(f"  Samples:     {len(X)}")
    print(f"  Loss:        {loss:.6f}")
    print(f"  Accuracy:    {metrics['accuracy']:.2%}")
    print(f"  Precision:   {metrics['precision']:.2%}")
    print(f"  Recall:      {metrics['recall']:.2%}")
    print(f"  Specificity: {metrics['specificity']:.2%}")
    print(f"  F1 score:    {metrics['f1']:.2%}")

    print_confusion_matrix(ann, X, Y, threshold)

    return metrics