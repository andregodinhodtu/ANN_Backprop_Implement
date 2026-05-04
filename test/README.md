# Testing

## Overview

Both implementations of the Artificial Neural Network, the base-Python version and the NumPy version,  share the same underlying structure, methods, and algorithms.

## Key Difference in Testing

The NumPy version takes advantage of vectorization: methods that handle only a single sample in base Python can process an entire batch in NumPy. Most methods behave identically across both versions — the NumPy version simply extends them to also accept batches. Testing is built around this principle: the NumPy implementation should be able to do everything the base-Python one does (and more), but not the other way around.

## Use of the function np.allclose()

Even in the base python testing, the function np.allclose() was used. The problem arose from the fact that pytest.approx works on flat sequences (and dicts and numpy arrays), but not on nested lists. That's the whole problem. The base python uses the nested lists to describe the vectors especially. So the procedure was to convert the nested lists to np.array and then assert if they are the same using np.allclose()
